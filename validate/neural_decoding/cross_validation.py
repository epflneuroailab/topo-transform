import numpy as np
import torch


class CrossValidation:
    def __init__(
        self,
        decoder,
        splits=(0.8, 0.1, 0.1),
        num_folds=5,
        stratified=False,
        contiguous_split=None,
    ):
        if stratified:
            raise NotImplementedError("Stratified cross-validation is not implemented yet.")

        self.decoder = decoder
        self.splits = splits
        self.num_folds = num_folds
        self.stratified = stratified
        self.contiguous_split = contiguous_split

    def __call__(self, X, y):
        all_scores = [[] for _ in range(len(self.splits))]
        for seed in range(self.num_folds):
            scores = self.validate(X, y, seed)
            for i, score in enumerate(scores):
                all_scores[i].append(score)
        all_scores = [np.array(scores).reshape(self.num_folds, -1) for scores in all_scores]
        return all_scores

    def validate(self, X, y, seed):
        splits = self._split(X, y, seed)
        self.decoder.fit(*splits[0])  # always fit on the first split (train)
        scores = []
        for split in splits[1:]:
            scores.append(self.decoder.score(*split))
        return scores

    def _split(self, X, y, seed):
        if self.contiguous_split is not None:
            num_blocks = len(X) // self.contiguous_split
            X = X[:num_blocks * self.contiguous_split].reshape(num_blocks, self.contiguous_split, -1)
            y = y[:num_blocks * self.contiguous_split].reshape(num_blocks, self.contiguous_split, -1)

        np.random.seed(seed)
        perm = np.random.permutation(len(X))

        sizes = [int(len(X) * s) for s in self.splits]
        sizes[-1] = len(X) - sum(sizes[:-1])  # ensure all samples are used

        indices = []
        start = 0
        for size in sizes:
            indices.append(perm[start:start+size])
            start += size

        splits = [(X[idx], y[idx]) for idx in indices]

        if self.contiguous_split is not None:
            splits = [
                (split[0].reshape(-1, split[0].shape[-1]), split[1].reshape(-1, split[1].shape[-1]))
                for split in splits
            ]

        return splits

def make_cv(decoder, contiguous_split=None, n_folds=5, stratified=False, split_percentages=None):
    cv = CrossValidation(
        decoder,
        splits=split_percentages if split_percentages is not None else (0.8, 0.1, 0.1),
        num_folds=n_folds,
        stratified=stratified,
        contiguous_split=contiguous_split,
    )
    return cv