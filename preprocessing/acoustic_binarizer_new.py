"""
    item: one piece of data
    item_name: data id
    wav_fn: wave file path
    spk: dataset name
    ph_seq: phoneme sequence
    ph_dur: phoneme durations
"""
import csv
import json
import os
import shutil

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from basics.base_binarizer import BaseBinarizer
from basics.base_pe import BasePE
from modules.fastspeech.tts_modules import LengthRegulator
from modules.pe import initialize_pe
from modules.vocoders.registry import VOCODERS
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_breathiness_pyworld,
    get_energy_librosa,
    get_mel2ph_torch,
)
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.phoneme_utils import locate_dictionary

os.environ["OMP_NUM_THREADS"] = "1"
ACOUSTIC_ITEM_ATTRIBUTES = [
    'spk_id',
    'wav',
    'mel',
    'tokens',
    'mel2ph',
    'ph_dur',
    'tokens',
    'f0',
    'energy',
    'breathiness',
    'key_shift',
    'speed'
]

pitch_extractor: BasePE = None
energy_smooth: SinusoidalSmoothingConv1d = None
breathiness_smooth: SinusoidalSmoothingConv1d = None


class AcousticBinarizerNew(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=ACOUSTIC_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator()
        self.need_energy = hparams.get('use_energy_embed', False)
        self.need_breathiness = hparams.get('use_breathiness_embed', False)

    def load_meta_data(self, raw_data_dir, ph_map, ds_id, spk_id):
        meta_data_dict = {}
        if (raw_data_dir / 'transcriptions.csv').exists():
            for utterance_label in csv.DictReader(
                    open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf-8')
            ):
                item_name = utterance_label['name']
                temp_dict = {
                    'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                    'ph_seq': [ph_map.get(x, x) for x in utterance_label['ph_seq'].split()],
                    'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()],
                    'spk_id': spk_id,
                }
                assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                    f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        else:
            raise FileNotFoundError(
                f'transcriptions.csv not found in {raw_data_dir}. '
                'If this is a dataset with the old transcription format, please consider '
                'migrating it to the new format via the following command:\n'
                'python scripts/migrate.py txt <INPUT_TXT>'
            )
        self.items.update(meta_data_dict)

    def process(self):
        # load each dataset
        for ds_id, spk_id, data_dir, ph_map in zip(range(len(self.raw_data_dirs)), self.spk_ids, self.raw_data_dirs, self.ph_maps):
            self.load_meta_data(data_dir, ph_map, ds_id=ds_id, spk_id=spk_id)

        self.item_names = list(self.items.keys())
        self._train_item_names, self._valid_item_names = self.split_train_valid_set()

        self.binary_data_dir.mkdir(parents=True, exist_ok=True)

        # Copy spk_map and dictionary to binary data dir
        spk_map_fn = self.binary_data_dir / 'spk_map.json'
        with open(spk_map_fn, 'w', encoding='utf-8') as f:
            json.dump(self.spk_map, f)
        shutil.copy(locate_dictionary(), self.binary_data_dir / 'dictionary.txt')
        self.check_coverage()

        # Process valid set and train set
        try:
            self.process_dataset(
                'valid',
                num_workers=int(self.binarization_args['num_workers']),
            )
            self.process_dataset(
                'train',
                num_workers=int(self.binarization_args['num_workers']),
            )
        except KeyboardInterrupt:
            exit(-1)

    def process_dataset(self, prefix, num_workers=0, apply_augmentation=False):
        args = []
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=self.data_attrs)

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        reverse_spk_map = {v: k for k, v in self.spk_map.items()}
        total_sec = {k: 0.0 for k in self.spk_map}
        extra_info = {
            'lengths': {},
            'seconds': {},
            'spk_ids': {},
        }

        def proc(arg):
            item = self.process_item(prefix, *arg)
            if item is None:
                return None, None, None, None
            item_no = builder.add_item(item)
            return item_no, item['length'], item['spk_id'], item['seconds']

        try:
            for item_no, item_len, item_spk_id, item_sec in tqdm(
                Parallel(n_jobs=num_workers, prefer='threads', return_as='generator')(delayed(proc)(arg) for arg in args), total=len(args),
                ncols=100
            ):
                if item_no is None:
                    continue
                extra_info['lengths'][item_no] = item_len
                extra_info['seconds'][item_no] = round(item_sec, 3)
                extra_info['spk_ids'][item_no] = item_spk_id
                total_sec[reverse_spk_map[item_spk_id]] += item_sec
            extra_info['lengths'] = list(map(lambda x: x[1], sorted(extra_info['lengths'].items(), key=lambda x: x[0])))
            extra_info['seconds'] = list(map(lambda x: x[1], sorted(extra_info['seconds'].items(), key=lambda x: x[0])))
            extra_info['spk_ids'] = list(map(lambda x: x[1], sorted(extra_info['spk_ids'].items(), key=lambda x: x[0])))
            extra_info['sp_token'] = self.phone_encoder.encode(['SP'])[0]
            assert len(extra_info['lengths']) == len(extra_info['seconds']) == len(extra_info['spk_ids'])
            assert len(extra_info['lengths']) == len(args)
        except KeyboardInterrupt:
            builder.finalize()
            raise
        builder.finalize()
        if prefix == 'train':
            with open(self.binary_data_dir / f'{prefix}.json', 'w') as f:
                # noinspection PyTypeChecker
                json.dump(extra_info, f)
        elif prefix == 'valid':
            with open(self.binary_data_dir / f'{prefix}.lengths', 'wb') as f:
                # noinspection PyTypeChecker
                np.save(f, extra_info['lengths'])
        else:
            raise NotImplementedError

        ref_len = np.percentile(sorted(total_sec.values()), 80)
        print(f'| {prefix} total duration: {sum(total_sec.values()):.3f}s')
        for k, v in sorted(total_sec.items(), key=lambda x: x[1], reverse=True):
            if v < ref_len:
                print(f'|     {k}: {v:.3f}s ({ref_len / v:.2f}x)')
            else:
                print(f'|     {k}: {v:.3f}s')

    @torch.no_grad()
    def process_item(self, prefix, item_name, meta_data, binarization_args):
        global pitch_extractor, energy_smooth, breathiness_smooth
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(meta_data['wav_fn'])
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(meta_data['wav_fn'])
        length = mel.shape[0]
        seconds = length * hparams['hop_size'] / hparams['audio_sample_rate']
        if prefix == 'train':
            processed_input = {
                'name': item_name,
                'wav_fn': meta_data['wav_fn'],
                'spk_id': meta_data['spk_id'],
                'length': len(wav),
                'seconds': len(wav) / hparams['audio_sample_rate'],
                'wav': wav,
                'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
                'ph_dur': np.array(meta_data['ph_dur']).astype(np.float32),
            }
            if pitch_extractor is None:
                pitch_extractor = initialize_pe()
            gt_f0, uv = pitch_extractor.get_pitch(
                wav, length, hparams, interp_uv=hparams['interp_uv']
            )
            if uv.all():  # All unvoiced
                print(f'Skipped \'{item_name}\': empty gt f0')
                return None
        elif prefix == 'valid':
            processed_input = {
                'name': item_name,
                'wav_fn': meta_data['wav_fn'],
                'spk_id': meta_data['spk_id'],
                'seconds': seconds,
                'length': length,
                'mel': mel,
                'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
                'ph_dur': np.array(meta_data['ph_dur']).astype(np.float32),
            }

            # get ground truth dur
            processed_input['mel2ph'] = get_mel2ph_torch(
                self.lr, torch.from_numpy(processed_input['ph_dur']), length, self.timestep, device=self.device
            ).cpu().numpy()

            # get ground truth f0
            if pitch_extractor is None:
                pitch_extractor = initialize_pe()
            gt_f0, uv = pitch_extractor.get_pitch(
                wav, length, hparams, interp_uv=hparams['interp_uv']
            )
            if uv.all():  # All unvoiced
                print(f'Skipped \'{item_name}\': empty gt f0')
                return None
            processed_input['f0'] = gt_f0.astype(np.float32)

            if self.need_energy:
                # get ground truth energy
                energy = get_energy_librosa(wav, length, hparams).astype(np.float32)

                if energy_smooth is None:
                    energy_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['energy_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                energy = energy_smooth(torch.from_numpy(energy).to(self.device)[None])[0]

                processed_input['energy'] = energy.cpu().numpy()

            if self.need_breathiness:
                # get ground truth breathiness
                breathiness = get_breathiness_pyworld(wav, gt_f0 * ~uv, length, hparams).astype(np.float32)

                if breathiness_smooth is None:
                    breathiness_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['breathiness_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0]

                processed_input['breathiness'] = breathiness.cpu().numpy()

            if hparams.get('use_key_shift_embed', False):
                processed_input['key_shift'] = 0.

            if hparams.get('use_speed_embed', False):
                processed_input['speed'] = 1.
        else:
            raise NotImplementedError

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
