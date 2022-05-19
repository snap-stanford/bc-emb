import torch
import argparse
import numpy as np
from tqdm import tqdm

def test_pinsage():
    from recsys.dataset import DynRecDataset
    from recsys.models import PinSAGE

    dataset = DynRecDataset('Amazon-Video_Games')

    torch.manual_seed(42)

    emb_dim = 64
    item_encoder = torch.nn.EmbeddingBag(dataset.num_item_attrs, emb_dim)
    model = PinSAGE(emb_dim, 2, item_encoder)

    model.eval()

    time = 0.5
    edge_index_useritem = dataset.edge_index_useritem(time)
    user = edge_index_useritem[0]
    item = edge_index_useritem[1]
    num_users = dataset.num_users(time)
    num_items = dataset.num_items(time)
    item_attr, item_attr_offset = dataset.item_attr_pair(time)
    
    model.refresh_all_embeddings(num_users, num_items, user, item, item_attr, item_attr_offset)
    print(model.x_user)
    print(model.x_item)

    num_users_init = num_users
    num_items_init = num_items

    time = 0.9
    edge_index_useritem = dataset.edge_index_useritem(time)
    user = edge_index_useritem[0]
    item = edge_index_useritem[1]
    num_users = dataset.num_users(time)
    num_items = dataset.num_items(time)
    item_attr, item_attr_offset = dataset.item_attr_pair(time)
    item_encoder = torch.nn.EmbeddingBag(dataset.num_item_attrs, emb_dim)

    model.refresh_all_embeddings(num_users, num_items, user, item, item_attr, item_attr_offset)
    print(model.x_user[:num_users_init])
    print(model.x_item[:num_items_init])

    users = torch.arange(100)
    pos = torch.arange(100)
    neg = torch.arange(100,200)
    items = torch.arange(100)

    print(model.get_users_scores(users))
    print(model.bpr_loss(users, pos, neg))
    print(model(users, items))

def test_pinsage_uniqueuser():
    from recsys.dataset import DynRecDataset
    from recsys.models import PinSAGE

    dataset = DynRecDataset('Amazon-Video_Games')

    torch.manual_seed(42)

    emb_dim = 64

    time = 0.5
    edge_index_useritem = dataset.edge_index_useritem(time)
    user = edge_index_useritem[0]
    item = edge_index_useritem[1]
    num_users = dataset.num_users(time)
    num_items = dataset.num_items(time)
    item_attr, item_attr_offset = dataset.item_attr_pair(time)

    item_encoder = torch.nn.EmbeddingBag(dataset.num_item_attrs, emb_dim)
    model = PinSAGE(emb_dim, 2, item_encoder, num_users)

    model.eval()
    
    model.refresh_all_embeddings(num_users, num_items, user, item, item_attr, item_attr_offset)
    print(model.x_user)
    print(model.x_item)

    print(model.x_user.shape)
    print(model.x_item.shape)
 

def test_rating_prediction():
    from recsys.dataset import DynRecDataset
    from recsys.models import BasicRatingPrediction
    from torch.optim import Adam

    parser = argparse.ArgumentParser(description='Basic Rating Prediction')
    parser.add_argument('--device', type=str, default='cpu',
                        help='which gpu to use if any (default: cpu)')
    parser.add_argument('--time', type=float, default=0.8,
                        help='which time to train the model on (default: 0.8)')
    parser.add_argument('--dataset', type=str, default='Amazon-Video_Games',
                        help='dataset name (default: Amazon-Video_Games)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='embedding dimensionality (default: 64)')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='embedding dimensionality (default: 16)')
    args = parser.parse_args()

    dataset = DynRecDataset('Amazon-Video_Games')
    time = args.time
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    edge_index_useritem = dataset.edge_index_useritem(time)
    user = edge_index_useritem[0]
    item = edge_index_useritem[1]
    ratings = dataset.edge_rating(time)
    num_users = dataset.num_users(time)
    num_items = dataset.num_items(time)
    num_edges = len(user)
    user_embeddings = torch.randn(num_users, emb_dim)
    item_embeddings = torch.randn(num_items, emb_dim)

    model = BasicRatingPrediction(emb_dim, hidden_dim).to(args.device)
    epochs = 50
    lr = 0.01
    batch_size = 100
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        sampled_indices = np.random.choice(range(num_edges), batch_size)
        users = user[sampled_indices]
        items = item[sampled_indices]
        sampled_user_embs = user_embeddings[users].to(args.device)
        sampled_item_embs = item_embeddings[items].to(args.device)
        pred_ratings = model(sampled_user_embs, sampled_item_embs)
        loss_fn = torch.nn.MSELoss()
        target_rating = ratings[sampled_indices].to(args.device)
        loss = loss_fn(pred_ratings, target_rating.view(-1,1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(pred_ratings)

def test_mlp():
    from recsys.models import MLP
    model = MLP(
        128,
        128,
        0,
        2,
        1,
    )

    x = torch.randn(10, 128)

    print('Without thresholding')
    model.train()
    y = model(x)
    print(y)

    model.eval()
    y = model(x)
    print(y)

    print('With thresholding')

    min_value = 0
    max_value = 0.1

    model = MLP(
        128,
        128,
        0,
        2,
        1,
        min_value = min_value,
        max_value = max_value,
    )

    model.train()
    y = model(x)
    print(y)

    model.eval()
    y = model(x)
    print(y)






if __name__ == '__main__':
    test_pinsage()
    test_pinsage_uniqueuser()
    test_rating_prediction()
    test_mlp()