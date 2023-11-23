from typing import Any
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.data = dataset

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        return self.data[index]
