import torch
import numpy as np

def collate_fn(batch):
    data, target, dataset_names = zip(*batch)
    data = torch.stack(data)
    target = torch.tensor(np.array(target))

    max_str_len = max(len(s) for s in dataset_names)
    encoded_names = torch.zeros(len(dataset_names), max_str_len, dtype=torch.uint8)

    for i, s in enumerate(dataset_names):
        encoded_names[i, :len(s)] = torch.tensor(list(s.encode('utf-8')), dtype=torch.uint8)

    return data, target, encoded_names

def collate_fn_with_index(batch):
    indices, batch = zip(*batch)
    data, target, encoded_names = collate_fn(batch)
    return indices, data, target, encoded_names