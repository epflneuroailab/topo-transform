import numpy as np
import torch

def _id(x):
    return x

def make_downsampler(method, target_size=None, seed=0):
    if method == 'random_gaussian':
        return RandomGaussian(target_size, seed)
    elif method == 'sparse_random_gaussian':
        return SparseRandomGaussian(target_size, seed)
    else:
        raise ValueError(f"Unknown downsampler type: {method}")


class RandomGaussian:
    __random_matrices__ = {}

    def __init__(self, target_size, random_seed, **kwargs):
        self.target_size = target_size
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)

    def __call__(self, x):
        x = x.reshape(x.size(0), -1)  # NOTE: also sample over time
        matrix_size = (x.size(1), self.target_size)
        if matrix_size not in self.__random_matrices__:
            random_matrix = torch.randn(*matrix_size, generator=self.generator)
            self.__random_matrices__[matrix_size] = random_matrix
        else:
            random_matrix = self.__random_matrices__[matrix_size]

        random_matrix = random_matrix.to(x.device)
        x = torch.mm(x, random_matrix)
        return x


class SparseRandomGaussian:
    __random_indices__ = {}
    __random_matrices__ = {}

    def __init__(self, target_size, random_seed, **kwargs):
        self.target_size = target_size
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)

    def __call__(self, x):
        x = x.reshape(x.size(0), -1)
        source_size = x.size(1)
        batch_size = x.size(0)
        projection_size = (source_size, self.target_size)

        # the value is set to the minimum density as recommended by Ping Li et al.: 1 / sqrt(n_features).
        density = 1 / np.sqrt(source_size)

        # use expected number of elements to speed up the process.
        source_sample_size = int(source_size * density)
        scaling = ((1 / density) / self.target_size) ** 0.5

        if projection_size not in self.__random_indices__:
            # sample with replacement to speed up the process.
            random_index = torch.randint(0, source_size, (source_sample_size, self.target_size), generator=self.generator)
            matrix_size = (source_sample_size, self.target_size)
            random_matrix = (torch.randint(0, 2, matrix_size, generator=self.generator) * 2 - 1) * scaling

            self.__random_indices__[projection_size] = random_index
            self.__random_matrices__[projection_size] = random_matrix
        else:
            random_index = self.__random_indices__[projection_size]
            random_matrix = self.__random_matrices__[projection_size]

        random_index = random_index.to(x.device)
        random_matrix = random_matrix.to(x.device)

        x_expanded = x.unsqueeze(1).expand(-1, source_sample_size, -1)
        expanded_index = random_index.unsqueeze(0).expand(batch_size, -1, -1)
        x_gathered = torch.gather(x_expanded, dim=2, index=expanded_index)
        # Sum over the sampled features (dim=1) to get shape (batch_size, target_size)
        x_projected = (x_gathered * random_matrix.unsqueeze(0)).sum(dim=1) 

        return x_projected