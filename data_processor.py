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
        self.label_name = []
        for file in os.listdir(data_path):
            category_name = file[:-8]   # category_name_png.npz
            data = np.load(os.path.join(data_path, file))[mode] # (size, 28, 28)
            label_no = int(label_dict[category_name])
            label = np.ones(np.shape(data)[0], dtype=int) * label_no
            X.append(data)
            y.append(label)
            self.label_name.append(category_name)

        self.X = np.concatenate(X)
        self.y = np.concatenate(y)
        print('Loaded %d samples from %s dataset.' % (self.__len__(), mode))

    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, idx):
        tensor_x = torch.tensor(self.X[idx]).float().unsqueeze(0)
        tensor_y = torch.tensor(self.y[idx]).long()
        return tensor_x, tensor_y
        

class RAWDataset(Dataset):
    def __init__(self, mode: str, data_path: str) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        X, y = [], []
        self.label_name = []
        for file in os.listdir(data_path):
            category_name = file[:-4]   # category_name.npz
            data = np.load(os.path.join(data_path, file), encoding='latin1', allow_pickle=True)[mode]   # (size, )
            label_no = int(label_dict[category_name])
            label = np.ones(np.shape(data)[0], dtype=int) * label_no
            X.append(data)
            y.append(label)
            self.label_name.append(category_name)

        self.X = np.concatenate(X)
        self.y = np.concatenate(y)
        print('Loaded %d samples from %s dataset.' % (self.__len__(), mode))

    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, idx):
        tensor_x = torch.tensor(self.X[idx]).float()    # (len, 3, )
        tensor_y = torch.tensor(self.y[idx]).long()     # (, )
        return tensor_x, tensor_y
        

class TwoBranchDataset(Dataset):
    def __init__(self, mode: str, data_path_raw: str, data_path_png: str) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        X_raw, X_png, y = [], [], []
        self.label_name = []
        for file_raw, file_png in zip(os.listdir(data_path_raw), os.listdir(data_path_png)):
            category_name_raw = file_raw[:-4]   # category_name.npz
            category_name_png = file_png[:-8]   # category_name_png.npz
            assert category_name_raw == category_name_png
            data_raw = np.load(os.path.join(data_path_raw, file_raw), encoding='latin1', allow_pickle=True)[mode]   # (size, )
            data_png = np.load(os.path.join(data_path_png, file_png))[mode]     # (size, 28, 28)
            label_no = int(label_dict[category_name_raw])
            label = np.ones(np.shape(data_raw)[0], dtype=int) * label_no
            X_raw.append(data_raw)
            X_png.append(data_png)
            y.append(label)
            self.label_name.append(category_name_raw)

        self.X_raw = np.concatenate(X_raw)
        self.X_png = np.concatenate(X_png)
        self.y = np.concatenate(y)
        print('Loaded %d samples from %s dataset.' % (self.__len__(), mode))

    def __len__(self):
        return np.shape(self.X_raw)[0]

    def __getitem__(self, idx):
        tensor_x_raw = torch.tensor(self.X_raw[idx]).float()    # (len, 3, )
        tensor_x_png = torch.tensor(self.X_png[idx]).float().unsqueeze(0)   # (28, 28)
        tensor_y = torch.tensor(self.y[idx]).long()     # (, )
        return tensor_x_raw, tensor_x_png, tensor_y
