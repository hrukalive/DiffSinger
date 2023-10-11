import csv
import json
import os
import pathlib
from collections import defaultdict
from random import choice, random

import librosa
import numpy as np
import torch
from tqdm import tqdm

from basics.base_binarizer import BaseBinarizer
from basics.base_pe import BasePE
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import SinusoidalSmoothingConv1d
from utils.hparams import hparams
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.plot import distribution_to_figure

os.environ["OMP_NUM_THREADS"] = "1"
DS_INDEX_SEP = '#'


class CheckLengthBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=[])

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        self.predict_variances = predict_energy or predict_breathiness
        self.lr = LengthRegulator().to(self.device)
        self.prefer_ds = self.binarization_args['prefer_ds']
        self.cached_ds = {}
        self.data_duplication = hparams['data_duplication']
        
        self.replacer = defaultdict(list)
        if 'ph_replacer' in hparams and hparams['ph_replacer']:
            with open(hparams['ph_replacer'], 'r') as f:
                for line in f:
                    line = line.strip().split()
                    for ph in line:
                        self.replacer[line[0]].append(ph)

    def load_attr_from_ds(self, ds_id, name, attr, idx=0):
        item_name = f'{ds_id}:{name}'
        item_name_with_idx = f'{item_name}{DS_INDEX_SEP}{idx}'
        if item_name_with_idx in self.cached_ds:
            ds = self.cached_ds[item_name_with_idx][0]
        elif item_name in self.cached_ds:
            ds = self.cached_ds[item_name][idx]
        else:
            ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}{DS_INDEX_SEP}{idx}.ds'
            if ds_path.exists():
                cache_key = item_name_with_idx
            else:
                ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}.ds'
                cache_key = item_name
            if not ds_path.exists():
                return None
            with open(ds_path, 'r', encoding='utf8') as f:
                ds = json.load(f)
            if not isinstance(ds, list):
                ds = [ds]
            self.cached_ds[cache_key] = ds
            ds = ds[idx]
        return ds.get(attr)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ph_map, ds_id, spk_id):
        meta_data_dict = {}
        dup = self.data_duplication[self.speakers[ds_id]]

        with open(raw_data_dir / 'transcriptions_filtered.csv', 'r', encoding='utf8') as f:
            for utterance_label in csv.DictReader(f):
                utterance_label: dict
                item_name = utterance_label['name']
                item_idx = int(item_name.rsplit(DS_INDEX_SEP, maxsplit=1)[-1]) if DS_INDEX_SEP in item_name else 0

                def require(attr):
                    if self.prefer_ds:
                        value = self.load_attr_from_ds(ds_id, item_name, attr, item_idx)
                    else:
                        value = None
                    if value is None:
                        value = utterance_label.get(attr)
                    if value is None:
                        raise ValueError(f'Missing required attribute {attr} of item \'{item_name}\'.')
                    return value

                dup_cnt = dup
                while random() < dup_cnt:
                    dup_cnt -= 1
                    rand_seq = []
                    for ph in [ph_map.get(x, x) for x in require('ph_seq').split()]:
                        if ph in self.replacer:
                            rand_seq.append(choice(self.replacer[ph]))
                        else:
                            rand_seq.append(ph)
                    temp_dict = {
                        'ds_idx': item_idx,
                        'spk_id': spk_id,
                        'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                        'ph_seq': rand_seq,
                        'ph_dur': [float(x) for x in require('ph_dur').split()]
                    }

                    assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                        f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'

                    if hparams['predict_dur']:
                        temp_dict['ph_num'] = [int(x) for x in require('ph_num').split()]
                        assert len(temp_dict['ph_seq']) == sum(temp_dict['ph_num']), \
                            f'Sum of ph_num does not equal length of ph_seq in \'{item_name}\'.'

                    if hparams['predict_pitch']:
                        temp_dict['note_seq'] = require('note_seq').split()
                        temp_dict['note_dur'] = [float(x) for x in require('note_dur').split()]
                        assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                            f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
                        assert any([note != 'rest' for note in temp_dict['note_seq']]), \
                            f'All notes are rest in \'{item_name}\'.'

                    meta_data_dict[f'{ds_id}:{item_name}_{dup_cnt}'] = temp_dict
        self.items.update(meta_data_dict)

    def check_coverage(self):
        super().check_coverage()

    def process(self):
        # load each dataset
        for ds_id, spk_id, data_dir, ph_map in zip(range(len(self.raw_data_dirs)), self.spk_ids, self.raw_data_dirs, self.ph_maps):
            self.load_meta_data(data_dir, ph_map, ds_id=ds_id, spk_id=spk_id)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._valid_item_names = self.split_train_valid_set(self.item_names)

        try:
            self.process_dataset('valid')
            self.process_dataset('train')
        except KeyboardInterrupt:
            exit(-1)

    def process_dataset(self, prefix, num_workers=0, apply_augmentation=False):
        args = []

        for i, (item_name, meta_data) in enumerate(self.meta_data_iterator(prefix)):
            args.append([item_name, meta_data, self.binarization_args])

        reverse_spk_map = {v: k for k, v in self.spk_map.items()}
        total_sec = {k: 0.0 for k in self.spk_map}

        def postprocess(item):
            nonlocal total_sec
            if item is None:
                return
            total_sec[reverse_spk_map[item['spk_id']]] += item['seconds']

        try:
            # code for single cpu processing
            for a in tqdm(args, ncols=100):
                item = self.process_item(*a)
                postprocess(item)
        except KeyboardInterrupt:
            raise

        ref_len = np.percentile(sorted(total_sec.values()), 80)
        print(f'| {prefix} total duration: {sum(total_sec.values()):.3f}s')
        for k, v in total_sec.items():#sorted(total_sec.items(), key=lambda x: x[1], reverse=True):
            if v > 0 and v < ref_len:
                print(f'|     {k}: {v:.3f}s ({ref_len / v:.2f}x)')
            else:
                print(f'|     {k}: {v:.3f}s')

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'seconds': librosa.get_duration(filename=meta_data['wav_fn']),
        }
        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
