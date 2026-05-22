import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config


class _Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label, _ = self.dataset[idx]
        return data, label

# do no transform and return the last layer features
class Extractor:
    def __call__(self, model, inputs, do_transform=True, average_time=True):
        with torch.no_grad():
            output = model.model(inputs)  # vjepa forward pass
            B, L, C = output.shape
            output = output.reshape(B, -1, 14, 14, C)  # (B, T, H, W, C)
            output = output.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

            if average_time:
                output = output.mean(dim=1)  # average over time dimension

            if do_transform:
                transform = model.transform.transforms[-1]
                output = output.reshape(-1, C, 14, 14)  # (B*T, C, H, W)
                output = transform(output) 
                output = output.reshape(B, -1, C, 14, 14)
                if average_time:
                    output = output[:, 0]  # (B, C, H, W)

            return output