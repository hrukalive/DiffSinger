from math import ceil

import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.rank_zero import rank_zero_debug

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.aux_decoder import build_aux_loss
from modules.losses.diff_loss import DiffusionNoiseLoss
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from training.acoustic_dataset import AcousticTrainingDataset
from utils.hparams import hparams
from utils.plot import get_bitmap_size, figure_to_image, spec_to_figure

matplotlib.use('Agg')


class AcousticDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(AcousticDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)
        self.required_variances = {}  # key: variance name, value: padding value
        if hparams.get('use_energy_embed', False):
            self.required_variances['energy'] = 0.0
        if hparams.get('use_breathiness_embed', False):
            self.required_variances['breathiness'] = 0.0

        self.need_key_shift = hparams.get('use_key_shift_embed', False)
        self.need_speed = hparams.get('use_speed_embed', False)
        self.need_spk_id = hparams['use_spk_id']

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


class AcousticTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_train_cls = AcousticDataset#AcousticTrainingDataset
        self.dataset_valid_cls = AcousticDataset
        self.use_shallow_diffusion = hparams['use_shallow_diffusion']
        if self.use_shallow_diffusion:
            self.shallow_args = hparams['shallow_diffusion_args']
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']

        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.logged_gt_wav = set()
        self.required_variances = []
        if hparams.get('use_energy_embed', False):
            self.required_variances.append('energy')
        if hparams.get('use_breathiness_embed', False):
            self.required_variances.append('breathiness')

    def build_model(self):
        return DiffSingerAcoustic(
            vocab_size=len(self.phone_encoder),
            out_dims=hparams['audio_num_mel_bins']
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.use_shallow_diffusion:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']
        self.mel_loss = DiffusionNoiseLoss(loss_type=hparams['diff_loss_type'])

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        target = sample['mel']  # [B, T_s, M]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')

        if hparams['use_spk_id']:
            spk_embed_id = sample['spk_ids']
        else:
            spk_embed_id = None
        output: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph, f0=f0, **variances,
            key_shift=key_shift, speed=speed, spk_embed_id=spk_embed_id,
            gt_mel=target, infer=infer
        )

        if infer:
            return output
        else:
            losses = {}

            if output.aux_out is not None:
                aux_out = output.aux_out
                norm_gt = self.model.aux_decoder.norm_spec(target)
                aux_mel_loss = self.lambda_aux_mel_loss * self.aux_mel_loss(aux_out, norm_gt)
                losses['aux_mel_loss'] = aux_mel_loss

            if output.diff_out is not None:
                x_recon, x_noise = output.diff_out
                mel_loss = self.mel_loss(x_recon, x_noise, nonpadding=(mel2ph > 0).unsqueeze(-1).float())
                losses['mel_loss'] = mel_loss

            return losses

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)
        if self.train_dataset and hasattr(self.train_dataset, 'set_device'):
            self.train_dataset.set_device(self.device)


    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        valid_per_replica = ceil(num_valid_plots / self.num_replicas)
        max_mel_len = max(self.valid_dataset.metadata['mel'][:num_valid_plots])
        max_wav_len = max_mel_len * hparams['hop_size'] + hparams['win_size']
        img_shape = get_bitmap_size('spec')
        self.validation_results = {
            'idxs': -torch.ones(valid_per_replica, dtype=torch.long),
            'pred_mel_imgs': torch.zeros((valid_per_replica, *img_shape), dtype=torch.uint8),
        }
        if self.use_shallow_diffusion:
            self.validation_results['aux_mel_imgs'] = torch.zeros((valid_per_replica, *img_shape), dtype=torch.uint8)
        if self.use_vocoder:
            if self.use_shallow_diffusion:
                self.validation_results['aux_wavs'] = torch.zeros(valid_per_replica, max_wav_len)
                self.validation_results['aux_wav_lens'] = torch.zeros(valid_per_replica, dtype=torch.long)
            self.validation_results['pred_wavs'] = torch.zeros(valid_per_replica, max_wav_len)
            self.validation_results['pred_wav_lens'] = torch.zeros(valid_per_replica, dtype=torch.long)
        # Logging GT wav to tensorboard if not yet logged
        if self.trainer.is_global_zero:
            for idx in range(num_valid_plots):
                if idx not in self.logged_gt_wav:
                    sample = self.valid_dataset[idx]
                    gt_wav = self.vocoder.spec2wav_torch(
                        sample['mel'].to(self.device).unsqueeze(0),
                        f0=sample['f0'].to(self.device).unsqueeze(0)
                    )
                    self.logger.experiment.add_audio(
                        f'gt_{idx}',
                        gt_wav,
                        sample_rate=hparams['audio_sample_rate'],
                        global_step=self.global_step
                    )
                    self.logged_gt_wav.add(idx)
        self.trainer.strategy.barrier()


    def _validation_step(self, sample, batch_idx):
        data_idx_base = batch_idx * (self.num_replicas * self.val_batch_size) + self.global_rank
        losses = self.run_model(sample, infer=False)
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        if data_idx_base < num_valid_plots and self.trainer.state.stage is RunningStage.VALIDATING:
            mel_out: ShallowDiffusionOutput = self.run_model(sample, infer=True)
            for idx in range(sample['size']):
                data_idx = data_idx_base + idx * self.num_replicas
                if data_idx < num_valid_plots:
                    val_idx = idx + batch_idx * self.val_batch_size
                    self.validation_results['idxs'][val_idx] = data_idx
            # Plots
            self.plot_mel(sample, mel_out, data_idx_base, batch_idx)
            if self.use_vocoder:
                self.plot_wav(sample, mel_out, data_idx_base, batch_idx)
        return losses, sample['size']


    def _on_validation_epoch_end(self):
        if self.trainer.state.stage is RunningStage.VALIDATING:
            gathered = self.all_gather(self.validation_results)
            if self.trainer.is_global_zero:
                # Need to flatten when using multiple devices
                if self.num_replicas > 1:
                    gathered = {
                        k: v.transpose(0, 1).flatten(0, 1)
                        for k, v in gathered.items()
                    }
                # Iterate and upload to Tensorboard
                for i in range(len(gathered['idxs'])):
                    idx = gathered['idxs'][i]
                    if idx < 0:
                        continue
                    # Diff prediction
                    pred_mel_img = gathered['pred_mel_imgs'][i]
                    pred_wav_len = gathered['pred_wav_lens'][i]
                    pred_wav = gathered['pred_wavs'][i, :pred_wav_len]
                    self.logger.experiment.add_image(
                        f'diffmel_{idx}',
                        pred_mel_img,
                        self.global_step
                    )
                    if self.use_vocoder:
                        self.logger.experiment.add_audio(
                            f'pred_{idx}',
                            pred_wav,
                            sample_rate=hparams['audio_sample_rate'],
                            global_step=self.global_step
                        )
                    # Aux prediction
                    if self.use_shallow_diffusion:
                        aux_mel_img = gathered['aux_mel_imgs'][i]
                        aux_wav_len = gathered['aux_wav_lens'][i]
                        aux_wav = gathered['aux_wavs'][i, :aux_wav_len]
                        self.logger.experiment.add_image(
                            f'auxmel_{idx}',
                            aux_mel_img,
                            self.global_step
                        )
                        if self.use_vocoder:
                            self.logger.experiment.add_audio(
                                f'aux_{idx}',
                                aux_wav,
                                sample_rate=hparams['audio_sample_rate'],
                                global_step=self.global_step
                            )
            self.trainer.strategy.barrier()

    ############
    # validation plots
    ############
    def plot_wav(self, sample, mel_out, data_idx_base, batch_idx):
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        for idx in range(sample['size']):
            data_idx = data_idx_base + idx * self.num_replicas
            if data_idx < num_valid_plots:
                val_idx = idx + batch_idx * self.val_batch_size
                mel_len = sample['mel_lengths'][idx]
                f0 = sample['f0'][idx, :sample['f0_lengths'][idx]].unsqueeze(0)
                if self.use_shallow_diffusion:
                    aux_mel = mel_out.aux_out[idx, :mel_len].unsqueeze(0)
                    aux_wav = self.vocoder.spec2wav_torch(aux_mel, f0=f0)
                    aux_wav_len = aux_wav.shape[-1]
                    self.validation_results['aux_wavs'][val_idx, :aux_wav_len] = aux_wav
                    self.validation_results['aux_wav_lens'][val_idx] = aux_wav_len
                pred_mel = mel_out.diff_out[idx, :mel_len].unsqueeze(0)
                pred_wav = self.vocoder.spec2wav_torch(pred_mel, f0=f0)
                pred_wav_len = pred_wav.shape[-1]
                self.validation_results['pred_wavs'][val_idx, :pred_wav_len] = pred_wav
                self.validation_results['pred_wav_lens'][val_idx] = pred_wav_len

    def plot_mel(self, sample, mel_out, data_idx_base, batch_idx):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        if self.use_shallow_diffusion:
            aux_spec_cat = torch.cat([(mel_out.aux_out - sample['mel']).abs() + vmin, sample['mel'], mel_out.aux_out], -1)
        pred_spec_cat = torch.cat([(mel_out.diff_out - sample['mel']).abs() + vmin, sample['mel'], mel_out.diff_out], -1)
        for idx in range(sample['size']):
            data_idx = data_idx_base + idx * self.num_replicas
            if data_idx < num_valid_plots:
                val_idx = idx + batch_idx * self.val_batch_size
                mel_len = sample['mel_lengths'][idx]
                title_text = f"{self.valid_dataset.metadata['spk'][data_idx]} - {self.valid_dataset.metadata['name'][data_idx]}"
                if self.use_shallow_diffusion:
                    self.validation_results['aux_mel_imgs'][val_idx] = torch.tensor(
                        figure_to_image(spec_to_figure(aux_spec_cat[idx, :mel_len, :], vmin, vmax, f"{title_text} (Aux MEL)"))
                    )
                self.validation_results['pred_mel_imgs'][val_idx] = torch.tensor(
                    figure_to_image(spec_to_figure(pred_spec_cat[idx, :mel_len, :], vmin, vmax, f"{title_text} (Diffusion MEL)"))
                )
