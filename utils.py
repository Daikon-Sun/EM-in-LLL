from multiprocessing import Pool
from torch.utils.data import Dataset, Sampler
import csv
import numpy as np
import os
import random
import torch


from settings import label_offsets


def prepare_inputs(batch):
    input_ids, masks, labels = tuple(b.cuda() for b in batch)
    return batch[0].shape[0], input_ids, masks, labels

def pad_to_max_len(input_ids, masks=None):
    max_len = max(len(input_id) for input_id in input_ids)
    masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    return input_ids, masks


def dynamic_collate_fn(batch):
    labels, input_ids = list(zip(*batch))
    labels = torch.tensor([b[0] for b in batch], dtype=torch.long)
    input_ids, masks = pad_to_max_len(input_ids)
    return input_ids, masks, labels


class TextClassificationDataset(Dataset):
    def __init__(self, task, mode, args, tokenizer):

        self.task = task
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = tokenizer.max_len
        self.n_test = args.n_test
        self.n_train = args.n_train
        self.valid_ratio = args.valid_ratio

        self.data = []
        self.label_offset = label_offsets[task.split('/')[-1]]
        if self.mode == "test":
            self.fname = os.path.join(task, "test.csv")
        elif self.mode in ["train", "valid"]:
            self.fname = os.path.join(task, "train.csv")

        with open(self.fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                self.data.append(row)

        random.shuffle(self.data)

        if mode == "test":
            self.data = self.data[:self.n_test]
        elif mode == "valid":
            self.data = self.data[:int(self.n_train * self.valid_ratio)]
        elif mode == "train":
            self.data = self.data[int(self.n_train * self.valid_ratio): self.n_train]

        with Pool(args.n_workers) as pool:
            self.data = pool.map(self.map_csv, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_csv(self, row):
        context = '[CLS]' + ' '.join(row[1:])[:self.max_len-2] + '[SEP]'
        return (int(row[0]) + self.label_offset, self.tokenizer.encode(context))


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = len(dataset)

    def __iter__(self):
        max_len = 0
        batch = []
        for idx in np.random.randint(self.n_samples, size=(self.n_samples,), dtype=np.int32):
            if max(max_len, len(self.dataset[idx][1]))**1.17 * (len(batch) + 1) > self.batch_size:
                yield batch
                max_len = 0
                batch = []
            max_len = max(max_len, len(self.dataset[idx][1]))
            batch.append(idx)
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


