import json
import gzip
from tqdm import tqdm
import torch
import os
import numpy as np
from collections import Counter

processed_dir = f'files/processed'

def extract_meta_review(category = 'Video_Games'):
    '''
        Extracting meta-information about the products
    '''

    processed_dir = f'files/{category}/processed'
    raw_dir = f'files/{category}/raw'

    path = f'{raw_dir}/meta_{category}.json.gz'
    g = gzip.open(path, 'r')
    asin2meta = {}
    for l in tqdm(g):
        line = json.loads(l)
        meta = {}
        meta['asin'] = line['asin']
        meta['brand'] = line['brand']
        meta['category_list'] = line['category']
        meta['main_category'] = line['main_cat']
        meta['also_view'] = line['also_view']
        meta['also_buy'] = line['also_buy']
        meta['title'] = line['title']

        asin2meta[line['asin']] = meta

    os.makedirs(processed_dir, exist_ok = True)

    torch.save(asin2meta, os.path.join(processed_dir, 'asin2meta.pt'))

    path = f'{raw_dir}/{category}_5.json.gz'

    g = gzip.open(path, 'r')

    review_list = []
    i = 0

    for l in tqdm(g):
        line = json.loads(l)
        rating = line['overall']

        time = line['reviewTime']
        time = time.replace(',', '')
        splitted = time.split(' ')
        mon = splitted[0].zfill(2)
        day = splitted[1][:2].zfill(2)
        year = splitted[2]
        time = f'{year}{mon}{day}'

        asin = line['asin']
        user = line['reviewerID']

        review_list.append((user, asin, rating, time))

    torch.save(review_list, os.path.join(processed_dir, 'reviews.pt'))

def create_graph(category = 'Video_Games'):
    '''
        Mapping everything into index
    '''

    processed_dir = f'files/{category}/processed'

    asin2meta = torch.load(os.path.join(processed_dir, 'asin2meta.pt'))
    review_list = torch.load(os.path.join(processed_dir, 'reviews.pt'))
    asinset = asin2meta.keys()

    filtered_review_list = []
    for review in review_list:
        # make sure the all the items have meta information
        if review[1] in asinset:
            filtered_review_list.append(review)

    timestamp_list = np.array([int(review[3]) for review in filtered_review_list])
    # sort according to time
    time_sorted_idx = np.argsort(timestamp_list)
    timestamp_list = timestamp_list[time_sorted_idx]

    unmapped_user_list_tmp = [filtered_review_list[i][0] for i in time_sorted_idx]
    unmapped_item_list_tmp = [filtered_review_list[i][1] for i in time_sorted_idx]
    rating_list = np.array([review[2] for review in filtered_review_list])
    rating_list = rating_list[time_sorted_idx]

    unique_user_set = set(unmapped_user_list_tmp)
    unique_item_set = set(unmapped_item_list_tmp)

    # mapping used for indexing (tmp)
    unique_user_list_tmp = sorted(list(unique_user_set))
    unique_item_list_tmp = sorted(list(unique_item_set))

    user2idx_tmp = {user: idx for idx, user in enumerate(unique_user_list_tmp)}
    item2idx_tmp = {item: idx for idx, item in enumerate(unique_item_list_tmp)}
    mapped_user_list_tmp = np.array([user2idx_tmp[unmapped_user] for unmapped_user in unmapped_user_list_tmp])
    mapped_item_list_tmp = np.array([item2idx_tmp[unmapped_item] for unmapped_item in unmapped_item_list_tmp])


    # find the first appearance of user/item
    _, first_appearance_user = np.unique(mapped_user_list_tmp, return_index = True)
    user_idx_sorted_by_time = np.argsort(first_appearance_user)
    user_idx_remapping = np.zeros(len(unique_user_list_tmp), dtype=np.int32)
    user_idx_remapping[user_idx_sorted_by_time] = np.arange(len(unique_user_list_tmp))

    _, first_appearance_item = np.unique(mapped_item_list_tmp, return_index = True)
    item_idx_sorted_by_time = np.argsort(first_appearance_item)
    item_idx_remapping = np.zeros(len(unique_item_list_tmp), dtype=np.int32)
    item_idx_remapping[item_idx_sorted_by_time] = np.arange(len(unique_item_list_tmp))

    # remap everything based on the first appearances
    unique_user_list = [unique_user_list_tmp[i] for i in user_idx_sorted_by_time]
    unique_item_list = [unique_item_list_tmp[i] for i in item_idx_sorted_by_time]
    user2idx = {user: idx for idx, user in enumerate(unique_user_list)}
    item2idx = {item: idx for idx, item in enumerate(unique_item_list)}
    mapped_user_list = user_idx_remapping[mapped_user_list_tmp]
    mapped_item_list = item_idx_remapping[mapped_item_list_tmp]

    unique_itemname_list = [asin2meta[item]['title'] for item in unique_item_list]

    print('#Users: ', len(user2idx))
    print('#Items: ', len(item2idx))
    print('#Interactions: ', len(mapped_user_list))

    # process also-view and also-buy
    mapped_also_view_mat = []
    mapped_also_buy_mat = []

    unmapped_brand_list = [] # only a single brand is assigned per item
    unmapped_category_mat = [] # multiple categories may be assigned per item

    for item_idx, item in enumerate(unique_item_list):
        meta = asin2meta[item]
        unmapped_also_view_list = meta['also_view']
        unmapped_also_buy_list = meta['also_buy']

        for also_view_item in unmapped_also_view_list:
            if also_view_item in item2idx:
                mapped_also_view_mat.append([item_idx, item2idx[also_view_item]])

        for also_buy_item in unmapped_also_buy_list:
            if also_buy_item in item2idx:
                mapped_also_buy_mat.append([item_idx, item2idx[also_buy_item]])

        unmapped_brand_list.append(meta['brand'])

        filtered_category_list = list(filter(lambda x: '</span>' not in x, meta['category_list']))
        unmapped_category_mat.append(filtered_category_list)

    mapped_also_view_mat = np.array(mapped_also_view_mat) # (num_entries, 2)
    mapped_also_buy_mat = np.array(mapped_also_buy_mat) # (num_entries, 2)

    unmapped_category_mat_concat = []
    for unmapped_category_list in unmapped_category_mat:
        unmapped_category_mat_concat.extend(unmapped_category_list)

    freq_thresh = 5

    # mapping used for indexing
    cnt = Counter(unmapped_brand_list)
    unique_brand_list = []
    for brand, freq in cnt.most_common(10000):
        if freq >= freq_thresh:
            unique_brand_list.append(brand)

    cnt = Counter(unmapped_category_mat_concat)
    unique_category_list = []
    for category, freq in cnt.most_common(10000):
        if freq >= freq_thresh:
            unique_category_list.append(category)

    brand2idx = {brand: idx for idx, brand in enumerate(unique_brand_list)}
    category2idx = {category: idx for idx, category in enumerate(unique_category_list)}
    print('brand category')
    print(len(brand2idx))
    print(len(category2idx))

    mapped_brand_mat = []
    for item_idx, brand in enumerate(unmapped_brand_list):
        if brand in brand2idx:
            mapped_brand_mat.append([item_idx, brand2idx[brand]])

    mapped_category_mat = []
    for item_idx, category_list in enumerate(unmapped_category_mat):
        for category in category_list:
            if category in category2idx:
                mapped_category_mat.append([item_idx, category2idx[category]])

    mapped_brand_mat = np.array(mapped_brand_mat)
    mapped_category_mat = np.array(mapped_category_mat)

    data_dict = {}
    data_dict['user'] = mapped_user_list
    data_dict['item'] = mapped_item_list
    data_dict['timestamp'] = timestamp_list
    data_dict['rating'] = rating_list
    data_dict['also_buy'] = mapped_also_buy_mat # first col also_buy second col
    data_dict['also_view'] = mapped_also_view_mat # first col also_view second col
    data_dict['brand'] = mapped_brand_mat # first col item has brand second col
    data_dict['category'] = mapped_category_mat # first col item has category second col
    data_dict['num_users'] = len(unique_user_list)
    data_dict['num_items'] = len(unique_item_list)
    data_dict['num_brands'] = len(unique_brand_list)
    data_dict['num_categories'] = len(unique_category_list)

    mapping_dict = {}
    mapping_dict['user'] = unique_user_list
    mapping_dict['item'] = unique_item_list
    mapping_dict['itemname'] = unique_itemname_list
    mapping_dict['brand'] = unique_brand_list
    mapping_dict['category'] = unique_category_list

    torch.save(data_dict, os.path.join(processed_dir, 'data_dict.pt'))
    torch.save(mapping_dict, os.path.join(processed_dir, 'mapping_dict.pt'))



if __name__ == '__main__':
    category_list = [
        'Musical_Instruments',
        'Video_Games',
        'Grocery_and_Gourmet_Food',
    ]

    for category in category_list:
        print(f'Processing {category} ...')
        extract_meta_review(category)
        create_graph(category)
        print()