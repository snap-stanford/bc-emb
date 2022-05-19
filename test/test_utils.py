import torch
import numpy as np

def test_edge_index_diff():
    from recsys.utils import take_edge_index_diff
    edge_index_new = torch.randint(10, size = (2, 15))
    edge_index_new = torch.unique(edge_index_new, dim = 1)

    edge_index_diff = edge_index_new[:,10:]
    edge_index_old = edge_index_new[:, :10]

    edge_index_diff2, _ = take_edge_index_diff(edge_index_new, edge_index_old)

    print(edge_index_diff)
    print(edge_index_diff2)
    assert (edge_index_diff == edge_index_diff2).all()

def test_negsampling():
    from recsys.dataset import DynRecDataset
    from recsys.utils import RecDatasetNegsamling

    dataset = DynRecDataset('Amazon-Video_Games')

    time_train = 0.8
    edge_index_useritem_train = dataset.edge_index_useritem(time_train)
    user_train = edge_index_useritem_train[0]
    item_train = edge_index_useritem_train[1]
    num_users_train = dataset.num_users(time_train)
    num_items_train = dataset.num_items(time_train)

    negsample_dataset = RecDatasetNegsamling(user_train, item_train, num_users_train, num_items_train)
    negsample_dataset.sample_neg_batch()

    print(negsample_dataset[0])


def test_split_embevolve():
    from recsys.dataset import DynRecDataset
    from recsys.utils import split_dynrecdataset

    dataset = DynRecDataset('Amazon-Video_Games')

    edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict \
            = split_dynrecdataset(dataset, time_train = 0.5, time_val = 0.55)
    print('edge_index_useritem_dict')
    print(edge_index_useritem_dict)

    unique_user_train = torch.unique(edge_index_useritem_dict['train'][0]).numpy()
    unique_user_val = torch.unique(edge_index_useritem_dict['val'][0]).numpy()

    assert np.sum(~np.isin(unique_user_val, unique_user_train)) == 0

    print()
    print('num_users_dict')
    print(num_users_dict)
    print()
    print('num_items_dict')
    print(num_items_dict)
    print()
    print('additional_dict_dict')
    print(additional_dict_dict)
    print(additional_dict_dict['train']['avg_item_ratings'].shape)
    print(additional_dict_dict['val']['avg_item_ratings'].shape)
    print()

def test_binarize_between():
    from recsys.utils import binarize_between
    x = torch.Tensor([1,2,5,6,3])
    print(binarize_between(x, 1, 3))

if __name__ == '__main__':
    test_edge_index_diff()
    test_negsampling()
    test_split_embevolve()
    test_binarize_between()
