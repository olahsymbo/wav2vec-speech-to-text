import random

import torch

class DataSplitter:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        Custom implementation of a DataLoader.

        Args:
            dataset (Dataset): An object that implements __len__ and __getitem__.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset at the start of each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        """
        Return the iterator object.
        """
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0
        return self

    def __next__(self):
        """
        Return the next batch.
        """
        if self.current >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current:self.current + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        self.current += self.batch_size
        return inputs, labels

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataSplitter:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current:self.current + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        inputs, labels = zip(*batch)
        
        inputs = [i.squeeze(0) if i.dim() == 2 else i for i in inputs]  # each i: [T]
        inputs = torch.stack(inputs).unsqueeze(1)  # (B, 1, T)

        self.current += self.batch_size
        return inputs, labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
