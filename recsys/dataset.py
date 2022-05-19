import numpy as np
import torch
import os
import copy
from typing import Optional, Tuple, Dict
from torch_scatter import scatter_max

from recsys.path import AMAZON_DIR

AVAILABLE_DATASETS = [
                    'Amazon-Video_Games',
                    'Amazon-Musical_Instruments',
                    'Amazon-Grocery_and_Gourmet_Food',
                    ]

class DynRecDataset(object):
    def __init__(self, name):
        assert name in AVAILABLE_DATASETS, f'{name} is not available'

        # Amazon review data
        if 'Amazon' in name:
            amazon_category = name.split('-')[-1]
            self._load_amazon_dataset(amazon_category)


    def _load_amazon_dataset(self, name):
        processed_dir = os.path.join(AMAZON_DIR, name, 'processed')
        data_dict = torch.load(os.path.join(processed_dir, 'data_dict.pt'))

        # set num_users, num_items (total number)
        self._num_users, self._num_items = data_dict['num_users'], data_dict['num_items']

        # set edge_index_useritem
        useritem = np.stack([data_dict['user'], data_dict['item']])
        self._edge_index_useritem = torch.from_numpy(useritem).to(torch.long)

        # set edge_timestamp
        self._edge_timestamp = torch.from_numpy(data_dict['timestamp']).to(torch.long)
        self._edge_timestamp_ratio = np.linspace(0, 1, len(self._edge_timestamp))

        # set rating
        self._edge_rating = torch.from_numpy(data_dict['rating']).to(torch.float)

        #### set item_attr_dict
        self._item_attr_dict = {}
        self._item_attr_offset_dict = {}
        self.num_item_attrs_dict = {}

        ### item_attr for category
        self._item_attr_dict['category'] = torch.from_numpy(data_dict['category'][:,1])
        self.num_item_attrs_dict['category'] = data_dict['num_categories']
        unique_item, _item_attr_offset_tmp = np.unique(data_dict['category'][:,0], return_index = True)
        if unique_item[0] != 0:
            unique_item = np.insert(unique_item, 0, 0)
            _item_attr_offset_tmp = np.insert(_item_attr_offset_tmp, 0, 0)

        _item_attr_offset = [0]
        for i in range(1, len(unique_item)):
            _item_attr_offset.extend([_item_attr_offset_tmp[i]] * (unique_item[i] - unique_item[i-1]))

        if unique_item[-1] < self._num_items - 1:
            for i in range(self._num_items - unique_item[-1] - 1):
                _item_attr_offset.append(len(data_dict['category'][:,1]))

        self._item_attr_offset_dict['category'] = torch.from_numpy(np.array(_item_attr_offset))

        ### item_attr for brand
        self._item_attr_dict['brand'] = torch.from_numpy(data_dict['brand'][:,1])
        self.num_item_attrs_dict['brand'] = data_dict['num_brands']
        unique_item, _item_attr_offset_tmp = np.unique(data_dict['brand'][:,0], return_index = True)
        if unique_item[0] != 0:
            unique_item = np.insert(unique_item, 0, 0)
            _item_attr_offset_tmp = np.insert(_item_attr_offset_tmp, 0, 0)

        _item_attr_offset = [0]
        for i in range(1, len(unique_item)):
            _item_attr_offset.extend([_item_attr_offset_tmp[i]] * (unique_item[i] - unique_item[i-1]))

        if unique_item[-1] < self._num_items - 1:
            for i in range(self._num_items - unique_item[-1] - 1):
                _item_attr_offset.append(len(data_dict['brand'][:,1]))

        self._item_attr_offset_dict['brand'] = torch.from_numpy(np.array(_item_attr_offset))


    def edge_index_useritem(self, time: Optional[float] = None) -> torch.LongTensor:
        if time is None:
            return self._edge_index_useritem
        else:
            return self._edge_index_useritem[:, self._edge_timestamp_ratio <= time]

    def edge_timestamp(self, time: Optional[float] = None) -> torch.LongTensor:
        if time is None:
            return self._edge_timestamp
        else:
            return self._edge_timestamp[self._edge_timestamp_ratio <= time]

    def edge_rating(self, time: Optional[float] = None) -> torch.FloatTensor:
        if time is None:
            return self._edge_rating
        else:
            return self._edge_rating[self._edge_timestamp_ratio <= time]

    def num_users(self, time: Optional[float] = None) -> int:
        return int(self.edge_index_useritem(time)[0].max() + 1)

    def num_items(self, time: Optional[float] = None) -> int:
        return int(self.edge_index_useritem(time)[1].max() + 1)

    def item_attr_pair_dict(self, time: Optional[float] = None) -> Dict[str, Tuple[torch.LongTensor, torch.LongTensor]]:
        '''
            Return a disctionary of pairs of (item_attr, item_attr_offset).
            Consider all kinds of available attributes
            Useful as input to torch.nn.EmbeddingBag
        '''

        num_items = self.num_items(time)
        if time is None or num_items == self._num_items:
            item_attr_pair_dict = {}
            for item_attr_name in self._item_attr_dict.keys():
                item_attr = self._item_attr_dict[item_attr_name]
                item_attr_offset = self._item_attr_offset_dict[item_attr_name]
                item_attr_pair_dict[item_attr_name] = (item_attr, item_attr_offset)
            return item_attr_pair_dict
        else:
            item_attr_pair_dict = {}
            for item_attr_name in self._item_attr_dict.keys():
                item_attr_offset = self._item_attr_offset_dict[item_attr_name][:num_items]
                item_attr = self._item_attr_dict[item_attr_name][:self._item_attr_offset_dict[item_attr_name][num_items]]
                item_attr_pair_dict[item_attr_name] = (item_attr, item_attr_offset)
            return item_attr_pair_dict

if __name__ == "__main__":
    dataset = DynRecDataset("Amazon-Video_Games")
