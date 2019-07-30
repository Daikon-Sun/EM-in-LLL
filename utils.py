import csv
from torch.utils.data import Dataset
import os
import torch
from multiprocessing import Pool
import random
import logging
logger = logging.getLogger(__name__)
import datetime


from settings import parse_args, label_offsets


def prepare_inputs(batch, device):
    input_ids, masks, labels = tuple(b.to(device) for b in batch)
    return batch[0].shape[0], input_ids, masks, labels

def pad_to_max_len(input_ids, masks=None):
    max_len = max(len(input_id) for input_id in input_ids)
    masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    return input_ids, masks


def dynamic_collate_fn(batch):
    labels, input_ids = list(zip(*batch))
    labels = torch.tensor([b[0] for b in batch], dtype=torch.long)
    max_len = max(len(b[1]) for b in batch)
    input_ids, masks = pad_to_max_len(input_ids)
    return input_ids, masks, labels


class TextClassificationDataset(Dataset):
    def __init__(self, task, mode, args, tokenizer):

        logger.info("Start parsing {} {} data...".format(task, mode))
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

class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.3f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True
