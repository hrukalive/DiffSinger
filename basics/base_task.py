import pathlib
import shutil
from datetime import datetime

import matplotlib

from basics.base_model import CategorizedModule

matplotlib.use('Agg')

from utils.hparams import hparams, set_hparams
import random
import sys
import numpy as np
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger
from utils.phoneme_utils import locate_dictionary
from utils.pl_utils import LatestModelCheckpoint, BaseTrainer, data_loader, DDP
from torch import nn
import torch.utils.data
import utils
import logging
import os

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseTask(nn.Module):
    '''
        Base class for training tasks.
        1. *load_ckpt*:
            load checkpoint;
        2. *training_step*:
            record and log the loss;
        3. *optimizer_step*:
            run backwards step;
        4. *start*:
            load training configs, backup code, log to tensorboard, start training;
        5. *configure_ddp* and *init_ddp_connection*:
            start parallel training.

        Subclasses should define:
        1. *build_model*, *build_optimizer*, *build_scheduler*:
            how to build the model, the optimizer and the training scheduler;
        2. *_training_step*:
            one training step of the model;
        3. *validation_end* and *_validation_end*:
            postprocess the validation output.
    '''

    def __init__(self, *args, **kwargs):
        # dataset configs
        super(BaseTask, self).__init__(*args, **kwargs)
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.example_input_array = None

        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences

        self.model = None
        self.persistent_dataloader = None
        self.training_losses_meter = None

    ###########
    # Training, validation and testing
    ###########
    def build_model(self):
        raise NotImplementedError

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.scheduler.get_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def on_epoch_end(self):
        pass
        # loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        # print(f"\n==============\n "
        #       f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}"
        #       f"\n==============\n")

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: dict
        """
        raise NotImplementedError

    def _validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}"
              f"\n==============\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        return [optm]

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, batch_by_size=True, persistent=False):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
        else:
            batches = batch_sampler
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False,
                                           persistent_workers=persistent)

    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ###########
    # Running configuration
    ###########

    @classmethod
    def start(cls):
        set_hparams()
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        task = cls()
        work_dir = pathlib.Path(hparams['work_dir'])
        trainer = BaseTrainer(
            checkpoint_callback=LatestModelCheckpoint(
                filepath=work_dir,
                verbose=True,
                monitor='val_loss',
                mode='min',
                num_ckpt_keep=hparams['num_ckpt_keep'],
                permanent_ckpt_start=hparams.get('permanent_ckpt_start', 0),
                permanent_ckpt_interval=hparams.get('permanent_ckpt_interval', -1),
                save_best=hparams['save_best'],
                period=1 if hparams['save_ckpt'] else 100000
            ),
            logger=TensorBoardLogger(
                save_dir=str(work_dir),
                name='lightning_logs',
                version='lastest'
            ),
            gradient_clip_val=hparams['clip_grad_norm'],
            val_check_interval=hparams['val_check_interval'],
            row_log_interval=hparams['log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches']
        )
        if not hparams['infer']:  # train
            # copy_code = input(f'{hparams["save_codes"]} code backup? y/n: ') == 'y'
            copy_code = True  # backup code every time
            if copy_code:
                t = datetime.now().strftime('%Y%m%d%H%M%S')
                code_dir = work_dir.joinpath('codes').joinpath(str(t))
                code_dir.mkdir(exist_ok=True, parents=True)
                for c in hparams['save_codes']:
                    shutil.copytree(c, code_dir, dirs_exist_ok=True)
                print(f'| Copied codes to {code_dir}.')
            # Copy spk_map.json and dictionary.txt to work dir
            binary_dir = pathlib.Path(hparams['binary_data_dir'])
            spk_map = work_dir.joinpath('spk_map.json')
            spk_map_src = binary_dir.joinpath('spk_map.json')
            if not spk_map.exists() and spk_map_src.exists():
                shutil.copy(spk_map_src, spk_map)
                print(f'| Copied spk map to {spk_map}.')
            dictionary = work_dir.joinpath('dictionary.txt')
            dict_src = binary_dir.joinpath('dictionary.txt')
            if not dictionary.exists():
                if dict_src.exists():
                    shutil.copy(dict_src, dictionary)
                else:
                    shutil.copy(locate_dictionary(), dictionary)
                print(f'| Copied dictionary to {dictionary}.')

            trainer.checkpoint_callback.task = task
            trainer.fit(task)
        else:
            trainer.test(task)

    def configure_ddp(self, model, device_ids):
        model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        if dist.get_rank() != 0 and not hparams['debug']:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        return model

    def training_end(self, *args, **kwargs):
        return None

    def init_ddp_connection(self, proc_rank, world_size):
        set_hparams(print_hparams=False)
        # guarantees unique ports across jobs from same grid search
        default_port = 12910
        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        root_node = '127.0.0.2'
        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    @data_loader
    def train_dataloader(self):
        return None

    @data_loader
    def test_dataloader(self):
        return None

    @data_loader
    def val_dataloader(self):
        return None

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        if isinstance(self.model, CategorizedModule):
            checkpoint['category'] = self.model.category

    def on_sanity_check_start(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_pre_performance_check(self):
        pass

    def on_post_performance_check(self):
        pass

    def on_before_zero_grad(self, optimizer):
        pass

    def on_after_backward(self):
        pass

    def backward(self, loss, optimizer):
        loss.backward()

    def grad_norm(self, norm_type):
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    results['grad_{}_norm_{}'.format(norm_type, name)] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results
