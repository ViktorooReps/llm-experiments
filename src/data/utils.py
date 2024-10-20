from typing import Type

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from .base import BaseDatasetConfig
from .slimpajama import get_slimpajama, SlimpajamaDatasetConfig


def get_config_type(dataset_name: str) -> Type[BaseDatasetConfig]:
    if dataset_name == 'slimpajama':
        return SlimpajamaDatasetConfig
    else:
        raise NotImplementedError(f"Unknown dataset '{dataset_name}'")


def get_data(tokenizer: PreTrainedTokenizer, dataset_name: str, config: BaseDatasetConfig) -> dict[str, list[np.ndarray]]:
    """ Fetch the right dataset given by the dataset_name parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'validation', corresponding to the tokenized training and validation data. """
    if dataset_name == 'slimpajama':
        assert isinstance(config, get_config_type(dataset_name)), f'incorrect config type: {config.__class__.__name__}'
        return get_slimpajama(tokenizer, config=config)
    else:
        raise NotImplementedError(f"Unknown dataset '{dataset_name}'")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list[np.ndarray], sequence_length: int, cutoff: int | None = None):
        super().__init__()
        self.data = data
        self.chunk_sizes = np.array(list(map(len, data)))

        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        self.examples_per_chunk = self.chunk_sizes // sequence_length
        self.chunk_start_idx = np.roll(np.cumsum(self.examples_per_chunk), shift=1)
        self.chunk_start_idx[0] = 0

        self.sequence_length = sequence_length
        self.cutoff = cutoff

    @property
    def n_tokens(self) -> int:
        if self.cutoff is not None:
            return self.cutoff * self.sequence_length
        return sum(self.chunk_sizes)

    def _chunk_for(self, idx):
        first_invalid = np.where(self.chunk_start_idx > idx)[0][0]
        return first_invalid - 1

    def __len__(self):
        if self.cutoff is not None:
            return self.cutoff
        return self.examples_per_chunk.sum()

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        data_source = self.data[self._chunk_for(idx)]

        idx = idx * seq_length
        x = torch.from_numpy((data_source[idx:idx + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (data_source[idx + 1:idx + 1 + seq_length]).astype(np.int64)
        )
        return x, y


def get_dataloader(data, sequence_length, batch_size, seed=0, distributed_backend=None):
    """Create a DataLoader for the given data. If distributed_backend is provided and is truly
    distributed (world size > 1), the DataLoader will be created with a DistributedSampler that
    splits the data across the processes (in conjunction with DDP).
    Otherwise, use a RandomSampler with the specified seed.

    Returns both the dataloader and the sampler.
    """
    dataset = Dataset(data, sequence_length=sequence_length)
    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    return loader, sampler
