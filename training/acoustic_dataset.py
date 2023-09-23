import json
import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from augmentation.spec_stretch import SpectrogramStretchAugmentation
from basics.base_dataset import BaseDataset
from modules.fastspeech.tts_modules import LengthRegulator
from modules.pe import initialize_pe
from modules.vocoders.registry import VOCODERS
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_mel2ph_torch,
)
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class AcousticTrainingDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        self.prefix = prefix
        self.preload = preload
        self.device = 'cpu'
        self.seed = hparams['seed']
        self.data_dir = hparams['binary_data_dir']
        self.raw_data_dirs = hparams['raw_data_dir']

        self.spk_map = None
        self.spk_ids = hparams['spk_ids']
        self.speakers = hparams['speakers']
        assert isinstance(self.speakers, list), 'Speakers must be a list'
        assert len(self.speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'
        if len(self.spk_ids) == 0:
            self.spk_ids = list(range(len(self.raw_data_dirs)))
        else:
            assert len(self.spk_ids) == len(self.raw_data_dirs), \
                'Length of explicitly given spk_ids must equal the number of raw datasets.'
        assert max(self.spk_ids) < hparams['num_spk'], \
            f'Index in spk_id sequence {self.spk_ids} is out of range. All values should be smaller than num_spk.'
        self.spk_map = {}
        for spk_name, spk_id in zip(self.speakers, self.spk_ids):
            if spk_name in self.spk_map and self.spk_map[spk_name] != spk_id:
                raise ValueError(f'Invalid speaker ID assignment. Name \'{spk_name}\' is assigned '
                                 f'with different speaker IDs: {self.spk_map[spk_name]} and {spk_id}.')
            self.spk_map[spk_name] = spk_id
        
        self.dup_factors = {}
        self.aug_settings = {}
        for spk_name, spk_id in self.spk_map.items():
            if spk_name in hparams['data_duplication']:
                self.dup_factors[spk_id] = max(1, hparams['data_duplication'][spk_name])
            else:
                self.dup_factors[spk_id] = 1
            if spk_name in hparams['augmentation_args_overrides']:
                self.aug_settings[spk_id] = self._update_aug_args(hparams['augmentation_args'], hparams['augmentation_args_overrides'][spk_name])
            else:
                self.aug_settings[spk_id] = hparams['augmentation_args']
        
        max_spk_id = max(self.spk_ids)
        self.replace_spk_ids = {}
        for spk_id in sorted(self.aug_settings.keys()):
            if self.aug_settings[spk_id]['fixed_pitch_shifting']['enabled']:
                for target in self.aug_settings[spk_id]['fixed_pitch_shifting']['targets']:
                    max_spk_id += 1
                    self.replace_spk_ids[(spk_id, target)] = max_spk_id
        hparams['num_spk'] = max_spk_id + 1

        self.spk_data = {}
        with open(os.path.join(self.data_dir, f'{self.prefix}.json')) as f:
            self.extra_info = json.load(f)
            for data_idx, spk_id in enumerate(self.extra_info['spk_ids']):
                if spk_id not in self.spk_data:
                    self.spk_data[spk_id] = []
                self.spk_data[spk_id].append(data_idx)

        self._indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        if preload:
            self.indexed_ds = [self._indexed_ds[i] for i in range(len(self._indexed_ds))]
        else:
            self.indexed_ds = self._indexed_ds

        self.sampling_rate = hparams['audio_sample_rate']
        self.win_size = hparams['win_size']
        self.hop_size = hparams['hop_size']
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']

        self.pitch_extractor = initialize_pe()
        self.energy_smooth = SinusoidalSmoothingConv1d(
            round(hparams['energy_smooth_width'] / self.timestep)
        ).eval()
        self.breathiness_smooth = SinusoidalSmoothingConv1d(
            round(hparams['breathiness_smooth_width'] / self.timestep)
        ).eval()

        self.lr = LengthRegulator()
        self.need_energy = hparams.get('use_energy_embed', False)
        self.need_breathiness = hparams.get('use_breathiness_embed', False)

        self.required_variances = {}  # key: variance name, value: padding value
        if hparams.get('use_energy_embed', False):
            self.required_variances['energy'] = 0.0
        if hparams.get('use_breathiness_embed', False):
            self.required_variances['breathiness'] = 0.0

        self.need_key_shift = hparams.get('use_key_shift_embed', False)
        self.need_speed = hparams.get('use_speed_embed', False)
        self.need_spk_id = hparams['use_spk_id']
        self.aug_ins = SpectrogramStretchAugmentation(None, None, pe=self.pitch_extractor)

        if hparams['vocoder'] in VOCODERS:
            self.vocoder = VOCODERS[hparams['vocoder']]()
        else:
            self.vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        
        self.prev_epoch = -1
        self.set_epoch(0)

    def _stft_len(self, l, keyshift=0.0, speed=1.0):
        win_size   = self.win_size
        hop_length = self.hop_size

        factor = 2 ** (keyshift / 12)       
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        new_l = l + (win_size_new - hop_length_new) // 2 + (win_size_new - hop_length_new + 1) // 2
        return math.ceil((new_l - win_size_new) / hop_length_new)

    def _update_aug_args(self, aug_args, overrides):
        ret = deepcopy(aug_args)
        if 'random_pitch_shifting' in overrides:
            ret.update(overrides['random_pitch_shifting'])
        if 'fixed_pitch_shifting' in overrides:
            ret.update(overrides['fixed_pitch_shifting'])
        if 'random_time_stretching' in overrides:
            ret.update(overrides['random_time_stretching'])
        return ret
    
    def set_device(self, device):
        self.device = device
        self.energy_smooth.to(device)
        self.breathiness_smooth.to(device)

    def set_epoch(self, epoch):
        if self.prev_epoch == epoch + self.seed:
            return
        self.epoch = epoch
        self.items = []
        self.sizes = []

        min_sil = int(0.1 * self.sampling_rate)
        max_sil = int(0.5 * self.sampling_rate)

        rng = np.random.default_rng(self.seed + self.epoch)
        for spk_id, data_idxs in self.spk_data.items():
            int_dup = int(self.dup_factors[spk_id])
            frac_dup = self.dup_factors[spk_id] - int_dup
            dup_idxs = deepcopy(data_idxs) * int_dup
            if frac_dup > 0:
                dup_idxs.extend(rng.choice(data_idxs, size=int(len(data_idxs) * frac_dup), replace=False).tolist())

            sp_times = rng.uniform(min_sil, max_sil, size=len(dup_idxs) * 2)
            sp_enables = rng.choice([0, 1], size=len(dup_idxs) * 2, p=[0.5, 0.5])
            sps = (sp_times * sp_enables).astype(int).tolist()
            for j, data_idx in enumerate(dup_idxs):
                if self.extra_info['seconds'][data_idx] >= 15:
                    sps[j] = 0
                    sps[j + len(dup_idxs)] = 0
                item = {
                    'data_idx': data_idx,
                    'spk_id': spk_id,
                    'sp': (sps[j], sps[j + len(dup_idxs)]),
                    'aug': {}
                }
                self.items.append(item)
                self.sizes.append(self._stft_len(
                    self.extra_info['lengths'][item['data_idx']] + item['sp'][0] + item['sp'][1]
                ))
            
            aug_setting = self.aug_settings[spk_id]
            aug_list = []
            total_scale = 0
            if aug_setting['random_pitch_shifting']['enabled']:
                aug_args = aug_setting['random_pitch_shifting']
                key_shift_min, key_shift_max = aug_args['range']
                assert self.need_key_shift, \
                    'Random pitch shifting augmentation requires use_key_shift_embed == True.'
                assert key_shift_min < 0 < key_shift_max, \
                    'Random pitch shifting augmentation must have a range where min < 0 < max.'

                # aug_ins = SpectrogramStretchAugmentation(None, aug_args, pe=self.pitch_extractor)
                scale = aug_args['scale']
                aug_item_names = rng.choice(data_idxs, size=int(scale * len(data_idxs)))
                rands = rng.uniform(-1, 1, size=len(aug_item_names)).tolist()

                for j, aug_item_name in enumerate(aug_item_names):
                    rand = rands[j]
                    if rand < 0:
                        key_shift = key_shift_min * abs(rand)
                    else:
                        key_shift = key_shift_max * rand
                    aug_task = {
                        'data_idx': aug_item_name,
                        # 'func': aug_ins,
                        'kwargs': {'key_shift': key_shift}
                    }
                    aug_list.append(aug_task)

                total_scale += scale

            if aug_setting['fixed_pitch_shifting']['enabled']:
                aug_args = aug_setting['fixed_pitch_shifting']
                targets = aug_args['targets']
                scale = aug_args['scale']
                assert not aug_setting['random_pitch_shifting']['enabled'], \
                    'Fixed pitch shifting augmentation is not compatible with random pitch shifting.'
                assert len(targets) == len(set(targets)), \
                    'Fixed pitch shifting augmentation requires having no duplicate targets.'
                assert self.need_spk_id, 'Fixed pitch shifting augmentation requires use_spk_id == True.'
                assert scale < 1, 'Fixed pitch shifting augmentation requires scale < 1.'

                # aug_ins = SpectrogramStretchAugmentation(None, aug_args, pe=self.pitch_extractor)
                for target in targets:
                    aug_item_names = rng.choice(data_idxs, size=int(scale * len(data_idxs))).tolist()
                    for aug_item_name in aug_item_names:
                        replace_spk_id = self.replace_spk_ids[(spk_id, target)]
                        aug_task = {
                            'data_idx': aug_item_name,
                            # 'func': aug_ins,
                            'kwargs': {'key_shift': target, 'replace_spk_id': replace_spk_id}
                        }
                        aug_list.append(aug_task)

                total_scale += scale * len(targets)

            if aug_setting['random_time_stretching']['enabled']:
                aug_args = aug_setting['random_time_stretching']
                speed_min, speed_max = aug_args['range']
                domain = aug_args['domain']
                assert self.need_speed, \
                    'Random time stretching augmentation requires use_speed_embed == True.'
                assert 0 < speed_min < 1 < speed_max, \
                    'Random time stretching augmentation must have a range where 0 < min < 1 < max.'
                assert domain in ['log', 'linear'], 'domain must be \'log\' or \'linear\'.'

                # aug_ins = SpectrogramStretchAugmentation(None, aug_args, pe=self.pitch_extractor)
                scale = aug_args['scale']
                k_from_raw = int(scale / (1 + total_scale) * len(data_idxs))
                k_from_aug = int(total_scale * scale / (1 + total_scale) * len(data_idxs))
                k_mutate = int(total_scale * scale / (1 + scale) * len(data_idxs))
                aug_types = [0] * k_from_raw + [1] * k_from_aug + [2] * k_mutate
                aug_items = rng.choice(data_idxs, size=k_from_raw).tolist() + \
                            rng.choice(range(len(aug_list)), size=k_from_aug + k_mutate).tolist()
                rands = rng.uniform(0, 1, size=len(aug_items)).tolist()

                for j, (aug_type, aug_item) in enumerate(zip(aug_types, aug_items)):
                    if domain == 'log':
                        # Uniform distribution in log domain
                        speed = speed_min * (speed_max / speed_min) ** rands[j]
                    else:
                        # Uniform distribution in linear domain
                        rand = rands[j] * 2 - 1
                        speed = 1 + (speed_max - 1) * rand if rand >= 0 else 1 + (1 - speed_min) * rand
                    if aug_type == 0:
                        aug_task = {
                            'data_idx': aug_item,
                            # 'func': aug_ins,
                            'kwargs': {'speed': speed}
                        }
                        aug_list.append(aug_task)
                    elif aug_type == 1:
                        real_aug = aug_list[aug_item]
                        aug_task = {
                            'data_idx': real_aug['data_idx'],
                            # 'func': real_aug['func'],
                            'kwargs': deepcopy(real_aug['kwargs'])
                        }
                        aug_task['kwargs']['speed'] = speed
                        aug_list.append(aug_task)
                    elif aug_type == 2:
                        real_aug = aug_list[aug_item]
                        real_aug['kwargs']['speed'] = speed
                total_scale += scale
            
            aug_sp_times = rng.uniform(min_sil, max_sil, size=len(aug_list) * 2)
            aug_sp_enables = rng.choice([0, 1], size=len(aug_list) * 2, p=[0.5, 0.5])
            aug_sps = (aug_sp_times * aug_sp_enables).astype(int).tolist()
            for j, aug in enumerate(aug_list):
                key_shift = aug['kwargs'].get('key_shift', 0.0)
                speed = aug['kwargs'].get('speed', 1.0)
                if self.extra_info['seconds'][data_idx] / speed >= 15:
                    aug_sps[j] = 0
                    aug_sps[j + len(aug_list)] = 0
                item = {
                    'data_idx': aug['data_idx'],
                    'spk_id': spk_id,
                    'sp': (aug_sps[j], aug_sps[j + len(aug_list)]),
                    'aug': {
                        # 'func': aug['func'],
                        'kwargs': aug['kwargs'],
                    }
                }
                self.items.append(item)
                self.sizes.append(self._stft_len(
                    self.extra_info['lengths'][item['data_idx']] + item['sp'][0] + item['sp'][1],
                    key_shift, speed
                ))
        self.prev_epoch = epoch + self.seed

    @torch.no_grad()
    def __getitem__(self, index):
        item = self.items[index]
        data = self.indexed_ds[item['data_idx']]

        wav_lst = []
        token_lst = []
        ph_dur_lst = []
        sp_l, sp_r = item['sp']
        total_sp = sp_l + sp_r
        if sp_l > 0:
            wav_lst.append(np.zeros((sp_l,)))
            token_lst.append([self.extra_info['sp_token']])
            ph_dur_lst.append([sp_l / self.sampling_rate])
        wav_lst.append(data['wav'])
        token_lst.append(data['tokens'])
        ph_dur_lst.append(data['ph_dur'])
        if sp_r > 0:
            wav_lst.append(np.zeros((sp_r,)))
            token_lst.append([self.extra_info['sp_token']])
            ph_dur_lst.append([sp_r / self.sampling_rate])
        wav = np.concatenate(wav_lst, dtype=np.float32)
        wav_torch = torch.from_numpy(wav).to(self.device)
        tokens = np.concatenate(token_lst, dtype=np.int64)
        ph_dur = np.concatenate(ph_dur_lst, dtype=np.float32)
        assert len(tokens) == len(ph_dur)

        mel = self.vocoder.wav2spec_direct(wav_torch, device=self.device)
        length = mel.shape[0]
        seconds = length * hparams['hop_size'] / hparams['audio_sample_rate']
        processed_input = {
            'spk_id': item['spk_id'],
            'seconds': seconds,
            'length': length,
            'mel': mel,
            'tokens': torch.from_numpy(tokens).to(self.device),
            'ph_dur': torch.from_numpy(ph_dur).to(self.device),
        }

        if self.need_energy:
            # get ground truth energy
            energy_data = data['energy']
            if total_sp > 0:
                energy_lst = []
                len_diff = length - energy_data.shape[0]
                len_l = int(round(sp_l / total_sp * len_diff))
                len_r = len_diff - len_l
                if len_l > 0:
                    energy_lst.append([-99.0] * len_l)
                energy_lst.append(energy_data)
                if len_r > 0:
                    energy_lst.append([-99.0] * len_r)
                energy = self.energy_smooth(torch.from_numpy(np.concatenate(energy_lst, dtype=np.float32)).to(self.device)[None])[0]
            else:
                energy = self.energy_smooth(energy_data.to(self.device)[None])[0]
            processed_input['energy'] = energy

        if self.need_breathiness:
            # get ground truth breathiness
            breathiness_data = data['breathiness']
            if total_sp > 0:
                breathiness_lst = []
                len_diff = length - breathiness_data.shape[0]
                len_l = int(round(sp_l / total_sp * len_diff))
                len_r = len_diff - len_l
                if len_l > 0:
                    breathiness_lst.append([-99.0] * len_l)
                breathiness_lst.append(breathiness_data)
                if len_r > 0:
                    breathiness_lst.append([-99.0] * len_r)
                breathiness = self.breathiness_smooth(torch.from_numpy(np.concatenate(breathiness_lst, dtype=np.float32)).to(self.device)[None])[0]
            else:
                breathiness = self.breathiness_smooth(breathiness_data.to(self.device)[None])[0]
            processed_input['breathiness'] = breathiness

        if not item['aug']:
            # get ground truth dur
            processed_input['mel2ph'] = get_mel2ph_torch(
                self.lr, processed_input['ph_dur'], length, self.timestep, device=self.device
            )

            # get ground truth f0
            f0_data = data['f0']
            if total_sp > 0:
                f0_lst = []
                len_diff = length - f0_data.shape[0]
                len_l = int(round(sp_l / total_sp * len_diff))
                len_r = len_diff - len_l
                if len_l > 0:
                    f0_lst.append([f0_data[0]] * len_l)
                f0_lst.append(f0_data)
                if len_r > 0:
                    f0_lst.append([f0_data[-1]] * len_r)
                processed_input['f0'] = torch.from_numpy(np.concatenate(f0_lst, dtype=np.float32)).to(self.device)
            else:
                processed_input['f0'] = f0_data.to(self.device)

            if hparams.get('use_key_shift_embed', False):
                processed_input['key_shift'] = 0.

            if hparams.get('use_speed_embed', False):
                processed_input['speed'] = 1.
        else:
            aug_out = self.aug_ins.process_item_wav(
                processed_input,
                self.vocoder,
                wav,
                wav_torch,
                **item['aug']['kwargs'],
                device=self.device
            )
            processed_input = aug_out
        
        return processed_input

    def collater(self, samples, max_len):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0, max_len)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0, max_len)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0, max_len)
        mel = utils.collate_nd([s['mel'] for s in samples], 0.0, max_len)
        batch.update({
            'tokens': tokens,
            'mel2ph': mel2ph,
            'mel': mel,
            'f0': f0,
            'mel_lengths': torch.LongTensor([s['mel'].shape[0] for s in samples]),
            'f0_lengths': torch.LongTensor([s['f0'].shape[0] for s in samples]),
        })
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad, max_len)
        if self.need_key_shift:
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])[:, None]
        if self.need_speed:
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])[:, None]
        if self.need_spk_id:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch
