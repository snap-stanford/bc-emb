from typing import List

import numpy as np
import torch
import random
from tqdm import tqdm
from collections import defaultdict

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min, scatter_add, scatter_std

# take difference of two edge_index
def take_edge_index_diff(edge_index_new: torch.LongTensor, edge_index_old: torch.LongTensor):
    '''
        Args:
            - edge_index_old: torch.LongTensor of shape (2, num_edges_old)
            - edge_index_new: torch.LongTensor of shape (2, num_edges_new)
        Return:
            - edge_index_diff: torch.LongTensor of shape (2, num_edges_diff)
                edge_index_new - edge_index_old
            - edge_diff_indices: np.array of indices in the new tensor which
                have been newly added.
    '''

    assert isinstance(edge_index_old, torch.LongTensor)
    assert isinstance(edge_index_new, torch.LongTensor)
    assert edge_index_old.shape[0] == 2
    assert edge_index_new.shape[0] == 2

    # convert to numpy
    edge_index_old_numpy = edge_index_old.cpu().numpy()
    edge_index_new_numpy = edge_index_new.cpu().numpy()

    # convert edge_index to 1d array
    max_elem = max(np.max(edge_index_old_numpy), np.max(edge_index_new_numpy)) + 1
    flattened_edge_index_old_numpy = edge_index_old_numpy[0] + edge_index_old_numpy[1] * max_elem
    flattened_edge_index_new_numpy = edge_index_new_numpy[0] + edge_index_new_numpy[1] * max_elem

    edge_diff_indices = ~np.isin(flattened_edge_index_new_numpy, flattened_edge_index_old_numpy)
    edge_index_diff_numpy = edge_index_new_numpy[:, edge_diff_indices]

    return torch.from_numpy(edge_index_diff_numpy).to(torch.long).to(edge_index_old.device), edge_diff_indices


def split_dynrecdataset(dynrecdataset, time_train: float, time_val: float):
    '''
        Split the DynRecDataset into chunks specified time_list
        - time_now
            Current timestamp
        - time_next
            Next timestamp
        - add_item_avg_rating: bool
            If True, adds information about the item average rating to the additionalDict.

        Return:
        - edge_index_useritem_dict
            - train
                training edges include the following
                - All edges until time_train
                - Edges between time_train and time_val that involve new users
                    - For new users with single interaction, include the interaction in training.
                    - For new users with K (>1) interactions, include the first K-1 interactions in training
            - val
                validation edges
                    - For existing users, include all edges between time_train and time_val
                    - For new users with  K (>1) interactions, include the last interaction in validation
        - num_users_dict
            - train
                existing users
            - val
                whole users
        - num_items_dict
            - train
                existing items
            - val
                whole items
        - additional_dict_dict
            - train
            - val
    '''

    assert time_train < time_val

    num_users_dict = {}
    num_items_dict = {}

    num_users_dict['train_original'] = dynrecdataset.num_users(time_train)
    num_items_dict['train_original'] = dynrecdataset.num_items(time_train)
    num_users_dict['val'] = dynrecdataset.num_users(time_val)
    num_items_dict['val'] = dynrecdataset.num_items(time_val)
    num_users_dict['train'] = num_users_dict['val']
    num_items_dict['train'] = num_items_dict['val']

    edge_index_useritem_until_time_train = dynrecdataset.edge_index_useritem(time_train)
    edge_rating_until_time_train = dynrecdataset.edge_rating(time_train)

    ## Mappening between time_train and time_val
    diff_idx = (time_train < dynrecdataset._edge_timestamp_ratio) & (dynrecdataset._edge_timestamp_ratio <= time_val)
    edge_index_useritem_diff = dynrecdataset._edge_index_useritem[:, diff_idx]
    user_diff = edge_index_useritem_diff[0]
    edge_timestamp_diff = dynrecdataset._edge_timestamp[diff_idx]
    edge_rating_diff = dynrecdataset._edge_rating[diff_idx]

    num_new_interactions_per_user = scatter_add(torch.ones(len(user_diff)), user_diff)
    num_new_interactions_per_new_user = num_new_interactions_per_user[num_users_dict['train_original']:]

    # index of new users with a single interaction between time_train and time_val
    new_user_idx_single_interaction = torch.nonzero(num_new_interactions_per_new_user == 1, as_tuple = True)[0] + num_users_dict['train_original']
    # index of new users with more than one interactions between time_train and time_val
    new_user_idx_few_interactions = torch.nonzero(num_new_interactions_per_new_user > 1, as_tuple = True)[0] + num_users_dict['train_original']

    # Extract the last appearance of users in user_diff
    unique_users_diff, unique_users_idx = np.unique(user_diff.numpy()[::-1], return_index = True)
    unique_users_idx = len(user_diff) - unique_users_idx - 1

    # Edges used for validation
    val_idx1 = torch.from_numpy(unique_users_idx[np.isin(unique_users_diff, new_user_idx_few_interactions.numpy())])
    val_idx2 = torch.nonzero(user_diff < num_users_dict['train_original'], as_tuple = True)[0]

    # user_diff is split into train and val parts
    val_idx = torch.sort(torch.cat([val_idx1, val_idx2])).values
    train_idx = torch.from_numpy(np.arange(len(user_diff))[~np.isin(np.arange(len(user_diff)), val_idx.numpy())])

    edge_index_useritem_dict = {}
    edge_index_useritem_dict['train'] = torch.cat([edge_index_useritem_until_time_train, edge_index_useritem_diff[:, train_idx]], dim = 1)
    edge_index_useritem_dict['val'] = edge_index_useritem_diff[:, val_idx]

    ###### Constructing additional dict
    additional_dict_dict = {}

    ##### training
    additional_dict_train = {}

    # edge rating
    additional_dict_train['edge_ratings'] = torch.cat([edge_rating_until_time_train, edge_rating_diff[train_idx]])
    
    # User interactions in validation set
    user_interactions = scatter_max(torch.ones(len(edge_index_useritem_dict['val'][0])), edge_index_useritem_dict['val'][0], dim_size = num_users_dict['train'])[0]
    user_interaction_max_ratings = scatter_max(edge_rating_diff[val_idx],  edge_index_useritem_dict['val'][0], dim_size = num_users_dict['train'])[0]
    user_interaction_min_ratings = scatter_min(edge_rating_diff[val_idx],  edge_index_useritem_dict['val'][0], dim_size = num_users_dict['train'])[0]
    user_interactions_num = scatter_add(torch.ones(len(edge_index_useritem_dict['val'][0])), edge_index_useritem_dict['val'][0], dim_size = num_users_dict['train'])

    # Item interactions in validation set
    item_interactions = scatter_max(torch.ones(len(edge_index_useritem_dict['val'][1])), edge_index_useritem_dict['val'][1], dim_size = num_items_dict['train'])[0]
    item_interaction_max_ratings = scatter_max(edge_rating_diff[val_idx],  edge_index_useritem_dict['val'][1], dim_size = num_items_dict['train'])[0]
    item_interaction_min_ratings = scatter_min(edge_rating_diff[val_idx],  edge_index_useritem_dict['val'][1], dim_size = num_items_dict['train'])[0]
    item_interactions_num = scatter_add(torch.ones(len(edge_index_useritem_dict['val'][1])), edge_index_useritem_dict['val'][1], dim_size = num_items_dict['train'])

    # avg item ratings
    thresh = 10 # only consider items with more than or equal to 10 user reviews
    unique_items_train = torch.unique(edge_index_useritem_dict['train'][1], sorted = True)
    avg_item_ratings_train = scatter_mean(additional_dict_train['edge_ratings'], edge_index_useritem_dict['train'][1])[unique_items_train]
    std_item_ratings_train = scatter_std(additional_dict_train['edge_ratings'], edge_index_useritem_dict['train'][1])[unique_items_train]
    item_idx_above_thresh = scatter_add(torch.ones(len(edge_index_useritem_dict['train'][1])), edge_index_useritem_dict['train'][1])[unique_items_train] >= thresh
    additional_dict_train['unique_items_for_avg_item_ratings'] = unique_items_train[item_idx_above_thresh]
    additional_dict_train['avg_item_ratings'] = avg_item_ratings_train[item_idx_above_thresh]
    additional_dict_train['std_item_ratings'] = std_item_ratings_train[item_idx_above_thresh]

    additional_dict_dict['train'] = additional_dict_train

    ##### validation
    additional_dict_val = {}
    additional_dict_val['edge_ratings'] = edge_rating_diff[val_idx]
    edge_index_useritem_trainval = torch.cat([edge_index_useritem_dict['train'], edge_index_useritem_dict['val']], dim = 1)
    edge_ratings_trainval = torch.cat([additional_dict_train['edge_ratings'], additional_dict_val['edge_ratings']])
    item_trainval = edge_index_useritem_trainval[1]

    unique_items_val = torch.unique(item_trainval, sorted = True)
    avg_item_ratings_val = scatter_mean(edge_ratings_trainval, item_trainval)[unique_items_val]
    std_item_ratings_val = scatter_std(edge_ratings_trainval, item_trainval)[unique_items_val]
    item_idx_above_thresh = scatter_add(torch.ones(len(item_trainval)), item_trainval)[unique_items_val] >= thresh
    additional_dict_val['unique_items_for_avg_item_ratings'] = unique_items_val[item_idx_above_thresh]
    additional_dict_val['avg_item_ratings'] = avg_item_ratings_val[item_idx_above_thresh]
    additional_dict_val['std_item_ratings'] = std_item_ratings_val[item_idx_above_thresh]
    additional_dict_val['user_activity'] = user_interactions
    additional_dict_val['user_activity_max_ratings'] = user_interaction_max_ratings
    additional_dict_val['user_activity_min_ratings'] = user_interaction_min_ratings
    additional_dict_val['user_activity_num'] = user_interactions_num
    additional_dict_val['item_activity'] = item_interactions
    additional_dict_val['item_activity_max_ratings'] = item_interaction_max_ratings
    additional_dict_val['item_activity_min_ratings'] = item_interaction_min_ratings
    additional_dict_val['item_activity_num'] = item_interactions_num

    additional_dict_dict['val'] = additional_dict_val

    return edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict   


def binarize(values, thresh):
    binarized_values = (values >= thresh).float()
    return binarized_values

def binarize_between(values, min_thresh, max_thresh):
    binarized_values = torch.logical_and(values >= min_thresh, values <= max_thresh).float()
    return binarized_values

class RecDatasetNegsamling(object):
    def __init__(self, user: torch.LongTensor, item: torch.LongTensor, num_users : int, num_items : int, user_uniform: bool = True):
        '''
            This object performs negative sampling for (user, item) transaction, where user is sampled uniformly.
            Therefore, each user is trained almost uniformly.

            - user: torch.LongTensor of shape (num_edges, )
            - item: torch.LongTensor of shape (num_edges, )
                user[i] interacts with item[i]

            - num_users: int
            - num_items: int
                number of users, items

            - user_uniform:
                If True, all the users are uniformly trained (regardless of the number of itneractions they make)
                This aligns better with the final evaluation metric we care about (e.g., Recall@K)

        '''
        self.num_users = num_users
        self.num_items = num_items
        self.user = user # (num_edges,)
        self.item = item # (num_edges,)
        self.num_edges = len(self.user)
        self.indices = np.arange(self.num_edges)

        # users are uniformly sampled
        self.uniform_user = None # (num_edges,)
        self.uniform_item = None # (num_edges,)

        # whether to do uniform sampling over users
        # if False, we do uniform sampling over edges
        self.user_uniform = user_uniform

        if self.user_uniform:
            unique_user, inverse, counts = np.unique(self.user, return_inverse=True, return_counts=True)
            counts[counts==0] = 1
            self.weights = 1./counts[inverse]

        super(RecDatasetNegsamling, self).__init__()

    def sample_neg_batch(self):
        '''
            Needs to be called at the beggning of each epoch to refresh the negative samples
        '''
        if self.user_uniform:
            uniform_idx = np.array(random.choices(self.indices, self.weights, k = self.num_edges))
        else:
            uniform_idx = np.array(random.choices(self.indices, k = self.num_edges))

        self.uniform_user = self.user[uniform_idx]
        self.uniform_item = self.item[uniform_idx]

        # [user, item]
        # all interactions
        idx_1 = self.user * self.num_items + self.item

        self.uniform_item_neg = torch.randint(self.num_items, (self.num_edges, ), dtype=torch.long)
        idx_2 = self.uniform_user * self.num_items + self.uniform_item_neg

        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        rest = mask.nonzero(as_tuple=False).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.randint(self.num_items, (rest.numel(), ), dtype=torch.long)
            idx_2 = self.uniform_user[rest] * self.num_items + tmp
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
            self.uniform_item_neg[rest] = tmp
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    def __getitem__(self, idx):
        return self.uniform_user[idx], self.uniform_item[idx], self.uniform_item_neg[idx]

    def __len__(self):
        return self.num_edges

class UserItemEdgeLevelDataset(object):
    def __init__(self, 
                    edge_index_useritem: torch.Tensor, 
                    edge_target: torch.Tensor,
                    user_emb: torch.Tensor,
                    item_emb: torch.Tensor,
                ):
        ''''
            Dataset object for an edge-level downstream prediction task
            Ex)
                - edge rating prediction

            - edge_index_useritem
                torch.Tensor of shape (2, num_edges)
                edge_index_useritem[0] represents user_index
                edge_index_useritem[1] represents item_index
            - edge_target
                torch.Tensor of shape (num_edges,)
            - user_emb:
                torch.Tensor of shape (num_users, dim)
            - item_emb:
                torch.Tensor of shape (num_items, dim)
        '''

        assert edge_index_useritem.shape[1] == len(edge_target)

        self.edge_index_useritem = edge_index_useritem
        self.edge_target = edge_target
        self.user_emb = user_emb
        self.item_emb = item_emb

        self.input_emb_dim = int(self.user_emb.shape[1] + self.item_emb.shape[1])

        super(UserItemEdgeLevelDataset, self).__init__()

    def __getitem__(self, idx):
        user_idx = self.edge_index_useritem[0, idx]
        item_idx = self.edge_index_useritem[1, idx]
        return torch.cat([self.user_emb[user_idx], self.item_emb[item_idx]]), self.edge_target[idx]

    def __len__(self):
        return len(self.edge_index_useritem[0])

class NodeLevelDataset(object):
    def __init__(self, 
                    node_target: torch.Tensor,
                    node_emb: torch.Tensor,
                ):
        ''''
            Dataset object for an node-level downstream prediction task
            Ex)
                - item rating prediction
                - item category prediction
                - user activity prediction

            - node_target
                torch.Tensor of shape (num_nodes,)
            - node_emb
                torch.Tensor of shape (num_nodes, dim)
        '''

        self.node_target = node_target
        self.node_emb = node_emb

        assert len(self.node_target) == len(self.node_emb)

        if self.node_emb.ndim == 2:
            self.input_emb_dim = int(self.node_emb.shape[1])

        super(NodeLevelDataset, self).__init__()

    def __getitem__(self, idx):
        return self.node_emb[idx], self.node_target[idx]

    def __len__(self):
        return len(self.node_target)