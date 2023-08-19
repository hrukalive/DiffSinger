import os

import numpy as np
from torch.utils.data import Dataset

from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class BaseDataset(Dataset):
    """
        Base class for datasets.
        1. *sizes*:
            clipped length if "max_frames" is set;
        2. *num_frames*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    """

    def __init__(self, prefix, preload):
        super().__init__()
        self.prefix = prefix
        self.data_dir = hparams['binary_data_dir']
        self.sizes = np.load(os.path.join(self.data_dir, f'{self.prefix}.lengths'))
        self._indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        if preload:
            self.indexed_ds = []
            for i in range(len(self._indexed_ds)):
                self.indexed_ds.append(self._indexed_ds[i])
            del self._indexed_ds
        else:
            self.indexed_ds = self._indexed_ds

    def __getitem__(self, index):
        return self.indexed_ds[index]

    def __len__(self):
        return len(self.sizes)

    def num_frames(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collater(self, samples):
        return {
            'size': len(samples)
        }
