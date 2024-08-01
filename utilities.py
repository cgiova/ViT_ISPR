import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


def permute(tensor):
    return tensor.permute(0, 3, 1,
                          2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)


def preprocess(data):
    tensor = torch.from_numpy(data).float()
    tensor = permute(tensor)
    return tensor


class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = preprocess(np.load(data_path))  # Preprocess and convert data to tensor
        self.labels = torch.from_numpy(np.load(labels_path)).long()  # Load labels and convert to tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Function to create DataLoader
def create_data_loader(data_path, labels_path, batch_size, shuffle=True):
    dataset = CustomDataset(data_path, labels_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
