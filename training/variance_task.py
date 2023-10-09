from math import ceil

import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data
from lightning.pytorch.trainer.states import RunningStage

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.losses.diff_loss import DiffusionNoiseLoss
from modules.losses.dur_loss import DurationLoss
from modules.metrics.curve import RawCurveAccuracy
from modules.metrics.duration import PhonemeDurationAccuracy, RhythmCorrectness
from modules.toplevel import DiffSingerVariance
from utils.hparams import hparams
from utils.plot import pitch_note_to_figure, curve_to_figure, dur_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(VarianceDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)
        need_energy = hparams['predict_energy']
        need_breathiness = hparams['predict_breathiness']
        self.predict_variances = need_energy or need_breathiness

    def collater(self, samples, max_len):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0, max_len)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0, max_len)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur,
            'tokens_lengths': torch.LongTensor([s['tokens'].shape[0] for s in samples]),
            'ph_dur_lengths': torch.LongTensor([s['ph_dur'].shape[0] for s in samples]),
        })

        if hparams['use_spk_id']:
            batch['spk_ids'] = torch.LongTensor([s['spk_id'] for s in samples])
        if hparams['predict_dur']:
            batch.update({
                'ph2word': utils.collate_nd([s['ph2word'] for s in samples], 0, max_len),
                'midi': utils.collate_nd([s['midi'] for s in samples], 0, max_len),
                'ph2word_lengths': torch.LongTensor([s['ph2word'].shape[0] for s in samples]),
                'midi_lengths': torch.LongTensor([s['midi'].shape[0] for s in samples]),
            })
        if hparams['predict_pitch']:
            if hparams['use_melody_encoder']:
                batch.update({
                    'note_midi': utils.collate_nd([s['note_midi'] for s in samples], -1),
                    'note_midi_lengths': torch.LongTensor([s['note_midi'].shape[0] for s in samples]),
                    'note_rest': utils.collate_nd([s['note_rest'] for s in samples], True),
                    'note_rest_lengths': torch.LongTensor([s['note_rest'].shape[0] for s in samples]),
                    'note_dur': utils.collate_nd([s['note_dur'] for s in samples], 0),
                    'note_dur_lengths': torch.LongTensor([s['note_dur'].shape[0] for s in samples]),
                    'mel2note': utils.collate_nd([s['mel2note'] for s in samples], 0),
                    'mel2note_lengths': torch.LongTensor([s['mel2note'].shape[0] for s in samples]),
                })
                if hparams['use_glide_embed']:
                    batch.update({
                        'note_glide': utils.collate_nd([s['note_glide'] for s in samples], 0),
                        'note_glide_lengths': torch.LongTensor([s['note_glide'].shape[0] for s in samples]),
                    })

            batch.update({
                'base_pitch': utils.collate_nd([s['base_pitch'] for s in samples], 0, max_len),
                'base_pitch_lengths': torch.LongTensor([s['base_pitch'].shape[0] for s in samples]),
            })
        if hparams['predict_pitch'] or self.predict_variances:
            batch.update({
                'mel2ph': utils.collate_nd([s['mel2ph'] for s in samples], 0, max_len),
                'pitch': utils.collate_nd([s['pitch'] for s in samples], 0, max_len),
                'uv': utils.collate_nd([s['uv'] for s in samples], True),
                'mel2ph_lengths': torch.LongTensor([s['mel2ph'].shape[0] for s in samples]),
                'pitch_lengths': torch.LongTensor([s['pitch'].shape[0] for s in samples]),
                'uv_lengths': torch.LongTensor([s['uv'].shape[0] for s in samples]),
            })
        if hparams['predict_energy']:
            batch.update({
                'energy': utils.collate_nd([s['energy'] for s in samples], 0, max_len),
                'energy_lengths': torch.LongTensor([s['energy'].shape[0] for s in samples]),
            })
        if hparams['predict_breathiness']:
            batch.update({
                'breathiness': utils.collate_nd([s['breathiness'] for s in samples], 0, max_len),
                'breathiness_lengths': torch.LongTensor([s['breathiness'].shape[0] for s in samples]),
            })

        return batch


def random_retake_masks(b, t, device):
    # 1/4 segments are True in average
    B_masks = torch.randint(low=0, high=4, size=(b, 1), dtype=torch.long, device=device) == 0
    # 1/3 frames are True in average
    T_masks = utils.random_continuous_masks(b, t, dim=1, device=device)
    # 1/4 segments and 1/2 frames are True in average (1/4 + 3/4 * 1/3 = 1/2)
    return B_masks | T_masks


class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_train_cls = VarianceDataset
        self.dataset_valid_cls = VarianceDataset

        self.use_spk_id = hparams['use_spk_id']

        self.predict_dur = hparams['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = hparams['lambda_dur_loss']

        self.predict_pitch = hparams['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        self.predict_variances = len(self.variance_prediction_list) > 0
        self.lambda_var_loss = hparams['lambda_var_loss']

    def build_model(self):
        return DiffSingerVariance(
            vocab_size=len(self.phone_encoder),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.predict_dur:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
            self.register_metric('rhythm_corr', RhythmCorrectness(tolerance=0.05))
            self.register_metric('ph_dur_acc', PhonemeDurationAccuracy(tolerance=0.2))
        if self.predict_pitch:
            self.pitch_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )
            self.register_metric('pitch_acc', RawCurveAccuracy(tolerance=0.5))
        if self.predict_variances:
            self.var_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )

    def run_model(self, sample, infer=False):
        spk_ids = sample['spk_ids'] if self.use_spk_id else None  # [B,]
        txt_tokens = sample['tokens']  # [B, T_ph]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        ph2word = sample.get('ph2word')  # [B, T_ph]
        midi = sample.get('midi')  # [B, T_ph]
        mel2ph = sample.get('mel2ph')  # [B, T_s]

        note_midi = sample.get('note_midi')  # [B, T_n]
        note_rest = sample.get('note_rest')  # [B, T_n]
        note_dur = sample.get('note_dur')  # [B, T_n]
        note_glide = sample.get('note_glide')  # [B, T_n]
        mel2note = sample.get('mel2note')  # [B, T_s]

        base_pitch = sample.get('base_pitch')  # [B, T_s]
        pitch = sample.get('pitch')  # [B, T_s]
        energy = sample.get('energy')  # [B, T_s]
        breathiness = sample.get('breathiness')  # [B, T_s]

        pitch_retake = variance_retake = None
        if (self.predict_pitch or self.predict_variances) and not infer:
            # randomly select continuous retaking regions
            b = sample['size']
            t = mel2ph.shape[1]
            device = mel2ph.device
            if self.predict_pitch:
                pitch_retake = random_retake_masks(b, t, device)
            if self.predict_variances:
                variance_retake = {
                    v_name: random_retake_masks(b, t, device)
                    for v_name in self.variance_prediction_list
                }

        output = self.model(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, mel2ph=mel2ph,
            note_midi=note_midi, note_rest=note_rest,
            note_dur=note_dur, note_glide=note_glide, mel2note=mel2note,
            base_pitch=base_pitch, pitch=pitch,
            energy=energy, breathiness=breathiness,
            pitch_retake=pitch_retake, variance_retake=variance_retake,
            spk_id=spk_ids, infer=infer
        )

        dur_pred, pitch_pred, variances_pred = output
        if infer:
            if dur_pred is not None:
                dur_pred = dur_pred.round().long()
            return dur_pred, pitch_pred, variances_pred  # Tensor, Tensor, Dict[str, Tensor]
        else:
            losses = {}
            if dur_pred is not None:
                losses['dur_loss'] = self.lambda_dur_loss * self.dur_loss(dur_pred, ph_dur, ph2word=ph2word)
            nonpadding = (mel2ph > 0).unsqueeze(-1) if mel2ph is not None else None
            if pitch_pred is not None:
                (pitch_x_recon, pitch_noise) = pitch_pred
                losses['pitch_loss'] = self.lambda_pitch_loss * self.pitch_loss(
                    pitch_x_recon, pitch_noise, nonpadding=nonpadding
                )
            if variances_pred is not None:
                (variance_x_recon, variance_noise) = variances_pred
                losses['var_loss'] = self.lambda_var_loss * self.var_loss(
                    variance_x_recon, variance_noise, nonpadding=nonpadding
                )
            return losses


    def _validation_step(self, sample, batch_idx):
        data_idx_base = batch_idx * (self.num_replicas * self.val_batch_size) + self.global_rank
        losses = self.run_model(sample, infer=False)
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        if data_idx_base < num_valid_plots and self.trainer.state.stage is RunningStage.VALIDATING:
            dur_preds, pitch_preds, variances_preds = self.run_model(sample, infer=True)
            ceil_idx = ceil((num_valid_plots - data_idx_base) / self.num_replicas)
            if dur_preds is not None:
                tokens = sample['tokens'][:ceil_idx]
                pred_durs = dur_preds[:ceil_idx]
                gt_durs = sample['ph_dur'][:ceil_idx]
                ph2words = sample['ph2word'][:ceil_idx]
                masks = tokens != 0
                self.rhythm_corr.update(pdur_pred=pred_durs, pdur_target=gt_durs, ph2word=ph2words, mask=masks)
                self.ph_dur_acc.update(pdur_pred=pred_durs, pdur_target=gt_durs, ph2word=ph2words, mask=masks)
                self.plot_dur(
                    data_idx_base, gt_durs, pred_durs, sample['ph_dur_lengths'][:ceil_idx],
                    tokens, sample['tokens_lengths'][:ceil_idx]
                )
            if pitch_preds is not None:
                pred_pitches = sample['base_pitch'][:ceil_idx] + pitch_preds[:ceil_idx]
                gt_pitches = sample['pitch'][:ceil_idx]
                masks = (sample['mel2ph'][:ceil_idx] > 0) & ~sample['uv'][:ceil_idx]
                self.pitch_acc.update(pred=pred_pitches, target=gt_pitches, mask=masks)
                self.plot_pitch(
                    data_idx_base, gt_pitches, pred_pitches, sample['pitch_lengths'][:ceil_idx],
                    sample['note_midi'][:ceil_idx], sample['note_dur'][:ceil_idx], sample['note_rest'][:ceil_idx],
                    sample['note_midi_lengths'][:ceil_idx], sample['note_dur_lengths'][:ceil_idx],
                    sample['note_rest_lengths'][:ceil_idx]
                )
            for name in self.variance_prediction_list:
                gt_variances = sample[name][:ceil_idx]
                pred_variances = variances_preds[name][:ceil_idx]
                self.plot_curve(
                    data_idx_base,
                    gt_curve=gt_variances,
                    pred_curve=pred_variances,
                    base_curves=None,
                    curve_lengths=sample[f'{name}_lengths'][:ceil_idx],
                    curve_name=name
                )
        return losses, sample['size']


    ############
    # validation plots
    ############
    def plot_dur(self, data_idx_base, gt_durs, pred_durs, dur_lenghts, txts, txt_lengths):
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        for idx in range(len(txts)):
            data_idx = data_idx_base + idx * self.num_replicas
            if data_idx < num_valid_plots:
                title_text = f"{self.valid_dataset.metadata['spk'][data_idx]} - {self.valid_dataset.metadata['name'][data_idx]}"
                gt_dur = gt_durs[idx][:dur_lenghts[idx]]
                pred_dur = pred_durs[idx][:dur_lenghts[idx]]
                txt = self.phone_encoder.decode(txts[idx][:txt_lengths[idx]]).split()
                self.logger.all_rank_experiment.add_figure(f'dur_{data_idx}', dur_to_figure(
                    gt_dur, pred_dur, txt, title_text
                ), self.global_step)

    def plot_pitch(self, data_idx_base,
        gt_pitches, pred_pitches, pitch_lenghts,
        note_midis, note_durs, note_rests,
        note_midi_lengths, note_dur_lengths, note_rest_lengths,
    ):
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        for idx in range(len(pitch_lenghts)):
            data_idx = data_idx_base + idx * self.num_replicas
            if data_idx < num_valid_plots:
                title_text = f"{self.valid_dataset.metadata['spk'][data_idx]} - {self.valid_dataset.metadata['name'][data_idx]}"
                gt_pitch = gt_pitches[idx][:pitch_lenghts[idx]]
                pred_pitch = pred_pitches[idx][:pitch_lenghts[idx]]
                note_midi = note_midis[idx][:note_midi_lengths[idx]]
                note_dur = note_durs[idx][:note_dur_lengths[idx]]
                note_rest = note_rests[idx][:note_rest_lengths[idx]]
                self.logger.all_rank_experiment.add_figure(f'pitch_{data_idx}', pitch_note_to_figure(
                    gt_pitch, pred_pitch, note_midi, note_dur, note_rest, title_text
                ), self.global_step)

    def plot_curve(self, data_idx_base, gt_curves, pred_curves, base_curves, curve_lengths, grid=None, curve_name='curve'):
        num_valid_plots = min(hparams['num_valid_plots'], len(self.valid_dataset))
        for idx in range(len(curve_lengths)):
            data_idx = data_idx_base + idx * self.num_replicas
            if data_idx < num_valid_plots:
                title_text = f"{self.valid_dataset.metadata['spk'][data_idx]} - {self.valid_dataset.metadata['name'][data_idx]}"
                gt_curve = gt_curves[idx][:curve_lengths[idx]]
                pred_curve = pred_curves[idx][:curve_lengths[idx]]
                if base_curves is not None:
                    base_curve = base_curves[idx][:curve_lengths[idx]]
                self.logger.all_rank_experiment.add_figure(f'{curve_name}_{data_idx}', curve_to_figure(
                    gt_curve, pred_curve, base_curve, grid=grid, title=title_text
                ), self.global_step)
