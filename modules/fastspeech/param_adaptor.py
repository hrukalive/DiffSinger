from __future__ import annotations

import torch

from modules.core.ddpm import MultiVarianceDiffusion
from utils import filter_kwargs

VARIANCE_CHECKLIST = ['energy', 'breathiness', 'voicing', 'tension']


class ParameterAdaptorModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variance_prediction_list = []
        self.predict_energy = config.get('predict_energy', False)
        self.predict_breathiness = config.get('predict_breathiness', False)
        self.predict_voicing = config.get('predict_voicing', False)
        self.predict_tension = config.get('predict_tension', False)
        if self.predict_energy:
            self.variance_prediction_list.append('energy')
        if self.predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        if self.predict_voicing:
            self.variance_prediction_list.append('voicing')
        if self.predict_tension:
            self.variance_prediction_list.append('tension')
        self.predict_variances = len(self.variance_prediction_list) > 0

    def build_adaptor(self, cls=MultiVarianceDiffusion):
        ranges = []
        clamps = []

        if self.predict_energy:
            ranges.append((
                self.config['energy_db_min'],
                self.config['energy_db_max']
            ))
            clamps.append((self.config['energy_db_min'], 0.))

        if self.predict_breathiness:
            ranges.append((
                self.config['breathiness_db_min'],
                self.config['breathiness_db_max']
            ))
            clamps.append((self.config['breathiness_db_min'], 0.))

        if self.predict_voicing:
            ranges.append((
                self.config['voicing_db_min'],
                self.config['voicing_db_max']
            ))
            clamps.append((self.config['voicing_db_min'], 0.))

        if self.predict_tension:
            ranges.append((
                self.config['tension_logit_min'],
                self.config['tension_logit_max']
            ))
            clamps.append((
                self.config['tension_logit_min'],
                self.config['tension_logit_max']
            ))

        variances_hparams = self.config['variances_prediction_args']
        total_repeat_bins = variances_hparams['total_repeat_bins']
        assert total_repeat_bins % len(self.variance_prediction_list) == 0, \
            f'Total number of repeat bins must be divisible by number of ' \
            f'variance parameters ({len(self.variance_prediction_list)}).'
        repeat_bins = total_repeat_bins // len(self.variance_prediction_list)
        kwargs = filter_kwargs(
            {
                'ranges': ranges,
                'clamps': clamps,
                'repeat_bins': repeat_bins,
                'timesteps': self.config.get('timesteps'),
                'time_scale_factor': self.config.get('time_scale_factor'),
                'backbone_type': self.config.get('backbone_type', self.config.get('diff_decoder_type')),
                'backbone_args': {
                    'n_layers': variances_hparams['residual_layers'],
                    'n_chans': variances_hparams['residual_channels'],
                    'n_dilates': variances_hparams['dilation_cycle_length'],
                }
            },
            cls
        )
        return cls(self.config, **kwargs)

    def collect_variance_inputs(self, **kwargs) -> list:
        return [kwargs.get(name) for name in self.variance_prediction_list]

    def collect_variance_outputs(self, variances: list | tuple) -> dict:
        return {
            name: pred
            for name, pred in zip(self.variance_prediction_list, variances)
        }
