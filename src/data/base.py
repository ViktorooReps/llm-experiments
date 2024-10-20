from dataclasses import dataclass


@dataclass
class BaseDatasetConfig:
    n_jobs: int = 40