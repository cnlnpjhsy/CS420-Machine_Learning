import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset


LABEL_PATH = "datasets/labels.json"
with open(LABEL_PATH, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)

def get_bounds(data):
    """Return bounds of data."""
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    for sample in data:
        x, y = sample
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return (min_x, max_x, min_y, max_y)


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
    def __init__(self, mode: str, data_path_raw: str, data_path_png: str, to_abs=False, remove_noise=False) -> None:
        super().__init__()
        self.to_abs = to_abs
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
            if remove_noise:
                index = self.select_unnoisy(data_png)
                X_raw.append(data_raw[index])
                X_png.append(data_png[index])
                y.append(label[index])
            else:
                X_raw.append(data_raw)
                X_png.append(data_png)
                y.append(label)
            self.label_name.append(category_name_raw)

        self.X_raw = np.concatenate(X_raw)
        self.X_png = np.concatenate(X_png)
        self.y = np.concatenate(y)
        self.png_size = np.shape(self.X_png)[1]
        print('Loaded %d samples from %s dataset.' % (self.__len__(), mode))
        if to_abs:
            X_abs = self.to_absolute()
            self.X_abs = np.array(X_abs, dtype=object)
            print('Vectors have been in absolute coordinates.')

    def __len__(self):
        return np.shape(self.X_raw)[0]

    def __getitem__(self, idx):
        tensor_x_raw = torch.tensor(self.X_raw[idx]).float()    # (len, 3, )
        tensor_x_png = torch.tensor(self.X_png[idx]).float().unsqueeze(0)   # (28, 28)
        tensor_y = torch.tensor(self.y[idx]).long()     # (, )
        if self.to_abs:
            tensor_x_abs = torch.tensor(self.X_abs[idx]).float()    # (len, 2, )
            return tensor_x_raw, tensor_x_abs, tensor_x_png, tensor_y
        return tensor_x_raw, tensor_x_png, tensor_y

    def to_absolute(self, margin=1.5):
        X_abs = []
        for idx, sample in enumerate(self.X_raw):
            # To absolute coordinates
            coordinates = sample[:, :2].astype(float)
            curr_x, curr_y = 0, 0
            for p in coordinates:
                p[0] += curr_x
                p[1] += curr_y
                curr_x = p[0]
                curr_y = p[1]
            # Centralize points
            min_x, max_x, min_y, max_y = get_bounds(coordinates)
            if max_x - min_x > max_y - min_y:
                norm = max_x - min_x
                border_y = (norm - (max_y - min_y)) * 0.5
                border_x = 0
            else:
                norm = max_y - min_y
                border_x = (norm - (max_x - min_x)) * 0.5
                border_y = 0
            # To png size
            norm = max(norm, 10e-6)
            scale = (self.png_size - 2 * margin) / norm
            dx = 0 - min_x + border_x
            dy = 0 - min_y + border_y
            coordinates += [dx, dy]
            coordinates *= scale
            coordinates += [margin, margin]
            # coordinates = np.round(coordinates).astype(np.int16)
            
            X_abs.append(coordinates)
        return X_abs

    def select_unnoisy(self, data_png, threshold=5):
        entropy = []
        for png in data_png:
            entropy.append(entropy_it(png.flatten()))
        entropy = np.array(entropy)
        lower = np.percentile(entropy, threshold)
        upper = np.percentile(entropy, 100 - threshold)
        index = np.where((entropy > lower) & (entropy < upper))
        return index


def entropy_it(x):
    counts = np.bincount(x)
    p = counts[counts > 0] / float(len(x))
    return -np.sum(p * np.log2(p))
