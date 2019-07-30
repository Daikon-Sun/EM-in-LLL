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
from utils import pad_to_max_len


class Memory:
    def __init__(self, args):
        self.n_neighbors = args.n_neighbors
        self.device = args.device
        with torch.no_grad():
            logger.info("Loading memory {} model".format(args.model_name))
            self.model = BertModel.from_pretrained(args.model_name)
            self.model.eval()
            self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.input_ids, self.masks, self.labels = [], [], [], []
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False


    # def pad_to_max_len(self, arr):
    #     return np.pad(arr, [(0, 0), (0, self.max_len - arr.shape[1])], 'constant')

    def add(self, input_ids, masks, labels):
        if len(self.labels) > 60:
            return
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        self.keys.extend(outputs[0][:, 0, :].detach().cpu().tolist())
        self.input_ids.extend(input_ids.cpu().tolist())
        self.masks.extend(masks.cpu().tolist())
        self.labels.extend(labels.cpu().tolist())
        del outputs


    def sample(self, n_samples):
        if self.built_tree:
            logging.warning("Tree already build! Ignore sample.")
            return
        inds = np.random.randint(len(self.labels), size=n_samples)
        input_ids = [self.input_ids[ind] for ind in inds]
        masks = [self.masks[ind] for ind in inds]
        labels = [self.labels[ind] for ind in inds]
        input_ids, masks = pad_to_max_len(input_ids, masks)
        labels = torch.tensor(labels, dtype=torch.long)
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
        input_ids, masks = list(zip(*[pad_to_max_len(input_id, mask) for input_id, mask in zip(self.input_ids[inds], self.masks[inds])]))
        labels = [torch.tensor(label, dtype=torch.long) for label in self.labels[inds]]
        return input_ids, masks, labels
