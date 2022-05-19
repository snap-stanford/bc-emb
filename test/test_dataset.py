def test_dynrecdataset_amazon():
    from recsys.dataset import DynRecDataset
    dataset = DynRecDataset('Amazon-Video_Games')

    print('edge_index_useritem')
    print(dataset.edge_index_useritem().shape)
    print(dataset.edge_index_useritem(1.0).shape)
    print(dataset.edge_index_useritem(0.5).shape)
    print()

    print('edge_timestamp')
    sliced_time_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    for time in sliced_time_list:
        print(time)
        print(dataset.edge_timestamp(time)[-1])
        print()

    print(dataset.edge_timestamp())
    print(dataset.edge_timestamp(1.0))
    print(dataset.edge_timestamp(0.5))
    print()

    print('edge_rating')
    print(dataset.edge_rating())
    print(dataset.edge_rating(1.0))
    print(dataset.edge_rating(0.5))
    print()

    print('num_users')
    print(dataset.num_users())
    print(dataset.num_users(1.0))
    print(dataset.num_users(0.5))
    print()

    print('num_items')
    print(dataset.num_items())
    print(dataset.num_items(1.0))
    print(dataset.num_items(0.5))
    print()

    print('item_attr_pair_dict')
    pair_dict = dataset.item_attr_pair_dict()
    print({item_attr_name: (item_attr.shape, item_attr_offset.shape) for item_attr_name, (item_attr, item_attr_offset) in pair_dict.items()})
    pair_dict = dataset.item_attr_pair_dict(1.0)
    print({item_attr_name: (item_attr.shape, item_attr_offset.shape) for item_attr_name, (item_attr, item_attr_offset) in pair_dict.items()})
    pair_dict = dataset.item_attr_pair_dict(0.5)
    print({item_attr_name: (item_attr.shape, item_attr_offset.shape) for item_attr_name, (item_attr, item_attr_offset) in pair_dict.items()})
    print()


def test_all_dataset():
    from recsys.dataset import DynRecDataset
    dataset_list =  [
        'Amazon-Video_Games',
        'Amazon-Musical_Instruments',
        'Amazon-Grocery_and_Gourmet_Food',
    ]

    for dataset_name in dataset_list:
        print(f'Dataset name: {dataset_name}')
        dataset = DynRecDataset(dataset_name)
        print(f'{dataset.num_users():,}')
        print(f'{dataset.num_items():,}')
        print(f'{dataset.edge_index_useritem().shape[1]:,}')
        print(f'{dataset.num_items():,}')
        print(dataset.num_item_attrs_dict)
        print()


if __name__ == '__main__':
    test_dynrecdataset_amazon()
    test_all_dataset()