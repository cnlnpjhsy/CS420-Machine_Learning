import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset


LABEL_PATH = "datasets/labels.json"
with open(LABEL_PATH, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)


class PNGDataset(Dataset):
    def __init__(self, mode: str, data_path: str) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        X, y = [], []
        for file in os.listdir(data_path):
            category_name = file[:-8]   # category_name_png.npz
            data = np.load(os.path.join(data_path, file))[mode] # (len, 28, 28)
            label_no = int(label_dict[category_name])
            label = np.ones(np.shape(data)[0], dtype=int) * label_no
            X.append(data)
            y.append(label)

        self.X = np.concatenate(X)
        self.y = np.concatenate(y)
        print('Loaded %d samples from %s dataset.' % (self.__len__(), mode))

    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, idx):
        tensor_x = torch.tensor(self.X[idx]).float().unsqueeze(0)
        tensor_y = torch.tensor(self.y[idx]).long()
        return tensor_x, tensor_y
        