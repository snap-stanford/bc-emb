
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from collections import defaultdict
from tqdm import tqdm

from recsys.utils import take_edge_index_diff

from sklearn.metrics import roc_auc_score

class RecallEvaluator:
    def __init__(self, edge_index_useritem_dict: Dict[str, torch.LongTensor], num_users_dict: Dict[str, int], num_items_dict: Dict[str, int]):
        '''
            Args:
            - edge_index_useritem_dict
            - num_users_dict
            - num_items_dict

            All have the three keys: 'train', 'val', and 'test'

        '''

        assert ('train' in edge_index_useritem_dict) and ('val' in edge_index_useritem_dict)
        assert ('train' in num_users_dict) and ('val' in num_users_dict)
        assert ('train' in num_items_dict) and ('val' in num_items_dict)

        self.split_list = []
        for split in ['train', 'val', 'test']:
            if split in edge_index_useritem_dict:
                self.split_list.append(split)

        # convert to numpy
        self.edge_index_useritem_dict_numpy = {}
        for key in edge_index_useritem_dict.keys():
            self.edge_index_useritem_dict_numpy[key] = edge_index_useritem_dict[key].cpu().numpy()

        self.num_users_dict = num_users_dict
        self.num_items_dict = num_items_dict

        self._build_index()

    def _build_index(self):
        print('Building indexing for evaluator...')

        '''
            maps split type into user2item
            user2item[i]: items liked by user i
        '''
        self.user2item_dict = {}

        for split, edge_index_useritem in self.edge_index_useritem_dict_numpy.items():
            user_split = edge_index_useritem[0]
            item_split = edge_index_useritem[1]
            data = np.ones(len(user_split))
            useritem_spmat = csr_matrix((data, (user_split, item_split)), shape = (self.num_users_dict[split], self.num_items_dict[split]))
            unique_user = np.unique(user_split)
            user2item = {}
            for u in tqdm(unique_user):
                user2item[u] = useritem_spmat[u].nonzero()[1]
            self.user2item_dict[split] = user2item

        '''
            maps split type into a set of users to evaluate the recall
        '''
        self.eval_users_dict = {}

        print('#Evaluation user ratio (ratio of users used for evaluation)')
        for split in self.split_list:
            self.eval_users_dict[split] = set(self.user2item_dict[split].keys())
            evaluser_ratio = len(self.eval_users_dict[split]) / float(self.num_users_dict[split])
            print(f"{split}: {evaluser_ratio}")
        print()

        print('New item ratio')
        for split in self.split_list:
            newitem_ratio = float(np.sum(self.edge_index_useritem_dict_numpy[split][1] >= self.num_items_dict['train'])) / len(self.edge_index_useritem_dict_numpy[split][1])
            print(f"{split}: {newitem_ratio}")

    def eval(self, scores: torch.Tensor, user_idx: torch.Tensor, split: str = 'test', K_list: List[int] = [10, 20, 50, 100]):
        '''
        Mini-batch-based evaluation for Recall@K
            args:
                scores: (num_users_batch, num_items), storing scoress predicted by the model
                user_idx: (num_users_batch, ), storing user indices in the mini-batch
                split: train, val, or test
                K: Recall@K
            return:
                - list of Recall@K scores
                - list of user indices for which predictio was made
        '''

        # check mode
        assert split in self.split_list

        # check shape
        num_users_batch = len(user_idx)
        assert scores.shape == (num_users_batch, self.num_items_dict[split])

        # extract users for which we evaluate performance
        eval_users_idx = []
        in_eval_users = []
        for u in list(map(int, user_idx)):
            tf = u in self.eval_users_dict[split]
            in_eval_users.append(tf)
            if tf:
                eval_users_idx.append(u)

        scores = scores[in_eval_users]
        eval_users_idx = np.array(eval_users_idx)

        # if there is no eval_user, just return empty list
        if len(eval_users_idx) == 0:
            recall_dict = {}
            for K in K_list:
                recall_dict[K] = np.array([])
            return eval_users_idx, recall_dict

        # what we are gonna predict
        groundtruth = [self.user2item_dict[split][u] for u in eval_users_idx]
        # what is already known to be positive
        allpos = []
        for u in eval_users_idx:
            if split == 'train':
                allpos.append([])
            elif split == 'val':
                if u in self.user2item_dict['train']:
                    allpos.append(self.user2item_dict['train'][u])
                else:
                    allpos.append([])
            elif split == 'test':
                if u in self.user2item_dict['train']:
                    base = [self.user2item_dict['train'][u]]
                else:
                    base = []
                if u in self.user2item_dict['val']:
                    base.append(self.user2item_dict['val'][u])

                if len(base) == 0:
                    allpos.append([])
                else:
                    allpos.append(np.hstack(base))

        # make sure the model never predicts the already known positives
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allpos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)

        scores[exclude_index, exclude_items] = -(1<<10)
        _, scores_K = torch.topk(scores, k=max(K_list))
        del scores
        scores_K = scores_K.cpu().numpy() # (num_eval_users, K)

        r = []
        for i in range(len(scores_K)):
            r.append(np.isin(scores_K[i], groundtruth[i]))
        r = np.stack(r).astype('float')
        recall_n = np.array([len(groundtruth[i]) for i in range(len(groundtruth))])

        recall_dict = {}
        for K in K_list:
            right_pred = np.sum(r[:,:K], axis = 1)
            recall = right_pred / recall_n
            recall_dict[K] = recall

        return eval_users_idx, recall_dict

class UnintendedTaskEvaluator:
    def __init__(self, metric: str):
        assert metric in ['binacc', 'rmse', 'rocauc', 'multiacc']
        self.metric = metric

        if metric in ['binacc', 'rocauc', 'multiacc']:
            self.better = 'higher'
        else:
            self.better = 'lower'

    def _eval_binacc(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape == y_true.shape
        y_pred_cls = y_pred > 0
        return float(torch.sum(y_pred_cls == y_true).item()) / len(y_pred)

    def _eval_rmse(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape == y_true.shape
        diff = y_pred - y_true
        return torch.sqrt(torch.mean(diff**2)).item()

    def _eval_rocauc(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape == y_true.shape
        y_pred_numpy = y_pred.cpu().numpy()
        y_true_numpy = y_true.cpu().numpy()
        return roc_auc_score(y_true_numpy, y_pred_numpy)

    def _eval_multiacc(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert len(y_pred) == len(y_true)
        y_pred_cls = torch.argmax(y_pred, dim = 1)
        return float(torch.sum(y_pred_cls == y_true).item()) / len(y_pred)

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.metric == 'binacc':
            return self._eval_binacc(y_pred, y_true)
        elif self.metric == 'rmse':
            return self._eval_rmse(y_pred, y_true)
        elif self.metric == 'rocauc':
            return self._eval_rocauc(y_pred, y_true)
        elif self.metric == 'multiacc':
            return self._eval_multiacc(y_pred, y_true)
