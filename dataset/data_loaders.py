import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class RecDataset(Dataset):
    def __init__(self, *args):
        self.users, self.items, self.ratings, self.tips = args

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.ratings[index], self.tips[index]


class RecDatasetManager(object):
    def __init__(self, opt):
        dataset_path = opt["dataset_path"]
        with open(dataset_path, 'rb') as fd:
            users, items, ratings, tips, user_ne_items, item_ne_users, \
                user_ne_users, item_ne_items, user_item_review = pickle.load(fd)
        opt["relations"] = (user_ne_items, item_ne_users, user_ne_users, item_ne_items, user_item_review)
        self.batch_size = opt["batch_size"]
        self.toolkits = opt["toolkits"]
        self.device = opt["device"]
        self.max_tip_len = opt["max_tip_len"]
        self.dataset = RecDataset(users, items, ratings, tips)
        self.dataloader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        users = []
        items = []
        ratings = []
        tips_vecs = []
        for u, i, s, t in batch:
            users.append(u)
            items.append(i)
            ratings.append(s)
            tips_vecs.append(self.toolkits.text2vec(t, add_end=True))
        tips_lens = np.array([len(t) for t in tips_vecs])
        tip_pad_len = min(max(tips_lens), self.max_tip_len)
        tips_pad_vecs = self.toolkits.pad2d(tips_vecs, tip_pad_len)

        tips_sorted_idx = np.argsort(tips_lens)[::-1]
        users = np.array(users)[tips_sorted_idx]
        items = np.array(items)[tips_sorted_idx]
        ratings = np.array(ratings)[tips_sorted_idx]
        tips_pad_vecs = tips_pad_vecs[tips_sorted_idx]

        return  (users, items, torch.from_numpy(ratings).to(self.device), torch.from_numpy(tips_pad_vecs).to(self.device))
        # pass
