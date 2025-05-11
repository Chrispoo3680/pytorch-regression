import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import polars as pl
from pathlib import Path
import os

from typing import Any, List, Union

NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    dataset: Any,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):

    # Split the dataset into a training and testing dataset (80% training, 20% testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Make the dataset into dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(train_data),
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_data),
    )

    return train_dataloader, test_dataloader, dataset


class CSVDataset(Dataset):
    def __init__(
        self,
        paths: List[Union[str, Path]],
        features_keys: List[str] = ["x"],
        target_keys: List[str] = ["y"],
    ):
        super().__init__()

        self.df = pl.scan_csv(paths, separator=",", schema_overrides={"x": float, "y": float}).collect().drop_nulls()  # type: ignore

        self.values = torch.tensor(
            list(zip(*[self.df[key].to_list() for key in features_keys])),
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            list(zip(*[self.df[key].to_list() for key in target_keys])),
            dtype=torch.float32,
        )

        self.samples = [(val, targ) for val, targ in zip(self.values, self.targets)]

    def __getitem__(self, index: int):

        sample = self.samples[index]

        return sample[0], sample[1]

    def __len__(self):
        return len(self.samples)


class LogisticFuncDataset(Dataset):
    def __init__(self, x_start=-10, x_end=10, steps=1000, C=10, a=1, b=1):
        super().__init__()

        self.values = torch.linspace(x_start, x_end, steps)
        self.labels = C / (1 + (torch.e ** (-b * self.values)))

    def __getitem__(self, index: int):
        return self.values[index], self.labels[index]

    def __len__(self):
        return len(self.values)
