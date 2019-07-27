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
from data_utils import prepare_inputs


class Memory:
    def __init__(self, args):
        self.model = BertModel.from_pretrained(args.model_name)
        self.model.eval()
        self.model.to(args.device)
        self.keys, self.input_ids, self.masks, self.labels = [], [], [], []
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False


    def add(self, input_ids, masks, labels):
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        self.keys.extend(outputs[0][:, 0, :].tolist())
        self.input_ids.extend(input_ids.cpu().tolist())
        self.masks.extend(masks.cpu().tolist())
        self.labels.extend(labels.cpu().tolist())

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

    def query(queries, n_neighbors):
        if not self.built_tree:
            logging.warning("Tree not built! Ignore query.")
            return
        queries = queries.cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=n_neighbors)
        return self.input_ids[inds], self.masks[inds], self.labels[inds]
