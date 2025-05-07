import torch
from torch.utils.data import random_split, DataLoader, Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, path: str, features_keys: list[str], target_keys: list[str]):
        super().__init__()

        df = pd.read_csv(path, sep=";")

        self.values = torch.tensor(
            zip(*[df[key].to_list() for key in features_keys]), dtype=torch.float32
        )
        self.targets = torch.tensor(
            zip(*[df[key].to_list() for key in target_keys]), dtype=torch.float32
        )

        self.samples = [(val, targ) for val, targ in zip(self.values, self.targets)]

    def __getitem__(self, index: int):
        return self.values[index], self.targets[index]

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
