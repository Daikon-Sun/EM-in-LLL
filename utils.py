import csv
from torch.utils.data import Dataset
import os
import re
import torch
from settings import parse_args
import logging
logger = logging.getLogger(__name__)
from multiprocessing import Pool


def dynamic_collate_fn(batch):
    labels = torch.tensor([b[0] for b in batch], dtype=torch.long)
    input_lens = [len(b[1]) for b in batch]
    max_len = max(input_lens)
    input_ids = torch.tensor([b[1] + [0] * (max_len - l) for b, l in zip(batch, input_lens)], dtype=torch.long)
    masks = torch.tensor([[1] * l + [0] * (max_len - l) for l in input_lens], dtype=torch.long)
    return input_ids, masks, labels


class TextClassificationDataset(Dataset):
    def __init__(self, root_dir, mode, args, tokenizer):

        logger.info("Parsing data...")

        self.root_dir = root_dir
        self.mode = mode
        self.tokenizer = tokenizer
        if self.mode == "test":
            self.num_test = args.num_test
            self.fname = os.path.join(root_dir, "test.csv")
        elif self.mode == "train" or self.mode == "valid":
            self.num_train = args.num_train
            self.valid_ratio = args.valid_ratio
            self.fname = os.path.join(root_dir, "train.csv")

        self.data = []
        with open(self.fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                self.data.append(row)
                if args.debug and len(self.data) >= 1000:
                    break

        with Pool(args.num_workers) as pool:
            self.data = pool.map(self.map_csv, self.data)

        self.num_labels = max(label for label, _ in self.data) + 1

        if self.mode in ["train", "valid"] and  self.num_train > len(self.data):
            logger.warning("number of data is less than args.num_train")
            self.num_train = args.num_train = len(self.data)
        if self.mode == "test" and self.num_test > len(self.data):
            logger.warning("number of data is less than args.num_test")
            self.num_test = args.num_test = len(self.data)

        if mode == "test":
            self.data = self.data[:self.num_test]
        elif mode == "valid":
            self.data = self.data[:int(self.num_train * self.valid_ratio)]
        elif mode == "train":
            self.data = self.data[int(self.num_train * self.valid_ratio): self.num_train]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_csv(self, row):
        context = '[CLS]' + ' '.join(row[1:]) + '[SEP]'
        return (int(row[0]) - 1, self.tokenizer.encode(context))

