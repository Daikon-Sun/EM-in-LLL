from pytorch_transformers import BertModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


from settings import model_classes
from utils import prepare_inputs


class Memory:
    def __init__(self, args):
        self.n_neighbors = args.n_neighbors
        self.device = args.device
        self.model = BertModel.from_pretrained(args.model_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.model.eval()
        self.model.to(args.device)
        self.keys = np.empty((0, self.hidden_size), dtype=np.int32)
        self.input_ids = np.empty((0, self.max_len), dtype=np.int32)
        self.masks = np.empty((0, self.max_len), dtype=np.int32)
        self.labels = np.array([], dtype=np.int32)
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False


    def pad_to_max_len(self, arr):
        return np.pad(arr, [(0, 0), (0, self.max_len - arr.shape[1])], 'constant')

    def add(self, input_ids, masks, labels):
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        self.keys = np.append(self.keys, outputs[0][:, 0, :].detach().cpu().numpy(), axis=0)
        self.input_ids = np.append(self.input_ids, self.pad_to_max_len(input_ids.cpu().numpy()), axis=0)
        self.masks = np.append(self.masks, self.pad_to_max_len(masks.cpu().numpy()), axis=0)
        self.labels = np.append(self.labels, labels.cpu().numpy(), axis=0)


    def sample(self, n_samples):
        inds = np.random.randint(len(self.labels), size=n_samples)
        input_ids = torch.tensor(self.input_ids[inds], dtype=torch.long)
        masks = torch.tensor(self.masks[inds], dtype=torch.long)
        labels = torch.tensor(self.labels[inds], dtype=torch.long)
        return input_ids.to(self.device), masks.to(self.device), labels.to(self.device)


    def build_tree(self):
        if self.built_tree:
            logging.warning("Tree already build! Ignore build.")
            return
        self.built_tree = True
        self.keys = np.array(self.keys)
        self.tree.fit(self.keys)
        self.input_ids = np.array(self.input_ids)
        self.masks = np.array(self.masks)
        self.labels = np.array(self.labels)


    def query(self, input_ids, masks):
        if not self.built_tree:
            logging.warning("Tree not built! Ignore query.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        queries = outputs[0][:, 0, :].cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=self.n_neighbors, return_distance=False)
        return self.input_ids[inds], self.masks[inds], self.labels[inds]
