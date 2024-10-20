import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os

from transformers import PreTrainedTokenizer, BatchEncoding

from src.data.base import BaseDatasetConfig

SPJ_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/slimpajama6B/")
SPJ_DATA_PATH_LARGE = os.path.join(os.path.dirname(__file__), "datasets/slimpajama_large/")
SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")

DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'datasets')


def frozenset_hash(frozen_vocab):
    # Convert frozenset to a sorted string (consistent ordering)
    vocab_str = ''.join(sorted(map(str, frozen_vocab)))
    return hashlib.md5(vocab_str.encode()).hexdigest()


@dataclass
class SlimpajamaDatasetConfig(BaseDatasetConfig):
    n_chunks: int = 1
    splits: str = 'train+validation'

    def get_splits(self) -> list[str]:
        return self.splits.split('+')


def get_slimpajama(
        tokenizer: PreTrainedTokenizer,
        config: SlimpajamaDatasetConfig = SlimpajamaDatasetConfig()
) -> dict[str, list[np.array]]:
    """
    Uses the tokenizer to encode the SlimPajama dataset and writes it in original chunks to disk. Returns the set
    of data sources for each split of the dataset.
    """
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

    if tokenizer.eos_token_id is None:
        raise ValueError(f"No EOS token found in the tokenizer")

    tokenizer_hash = frozenset_hash(frozenset(tokenizer.get_vocab().items()))

    datasets_path = Path(DATASETS_PATH)
    datasets_path.mkdir(exist_ok=True)

    spj_path = datasets_path.joinpath(f"slimpajama_tok-{tokenizer.__class__.__name__}-{tokenizer_hash}")
    spj_path.mkdir(exist_ok=True)

    def tokenize_batch(batch):
        tokenized_batch: BatchEncoding = tokenizer(
            batch["text"],
            padding=False,
            add_special_tokens=False
        )

        return {
            'ids': [ids + [tokenizer.eos_token_id] for ids in tokenized_batch["input_ids"]],
            'len': list(map(len, tokenized_batch["ids"]))
        }

    mmaped_chunks = {}

    for split in config.get_splits():
        split_dir = spj_path.joinpath(split)
        split_dir.mkdir(exist_ok=True)

        mmaped_chunks[split] = []

        for chunk in range(1, config.n_chunks + 1):
            chunk_path = split_dir.joinpath(f"chunk{chunk}.bin")

            if not chunk_path.exists():
                dataset = load_dataset("cerebras/SlimPajama-627B", split=f"{split}/chunk{chunk}")
                tokenized = dataset.map(  # noqa
                    tokenize_batch,
                    remove_columns=dataset.column_names,
                    desc=f"tokenizing {split}/chunk{chunk}",
                    batched=True,
                    num_proc=config.n_jobs,
                )

                arr_len = np.sum(tokenized["len"])
                total_batches = min(1024, len(tokenized))

                arr = np.memmap(chunk_path, dtype=dtype, mode="w+", shape=(arr_len,))

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f"writing {chunk_path}"):
                    # Batch together samples for faster write
                    batch = tokenized.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["ids"])
                    # Write into mmap
                    arr[idx: idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

            # load the chunk in read-only mmap

            mmaped_chunks[split].append(np.memmap(chunk_path, dtype=dtype, mode="r"))

    return mmaped_chunks
