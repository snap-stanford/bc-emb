import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from recsys.models import PinSAGE, ItemEncoder
from recsys.utils import RecDatasetNegsamling, split_dynrecdataset
from recsys.evaluate import RecallEvaluator
from recsys.dataset import DynRecDataset
from recsys.shared import sliced_time_list, get_pinsage_hyperparam_list

import os
from tqdm import tqdm
import numpy as np
import random
import copy
import shutil

def train(model, device, loader, optimizer, weight_decay, config_dict):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        user, item, item_neg = batch
        user, item, item_neg = user.to(device), item.to(device), item_neg.to(device)

        model.refresh_all_embeddings(
            config_dict['num_users'],
            config_dict['num_items'],
            config_dict['user'],
            config_dict['item'],
            config_dict['item_attr_pair_dict'],
        )

        loss, reg_loss = model.bpr_loss(user, item, item_neg)
        loss = loss + reg_loss * weight_decay

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

@torch.no_grad()
def eval(model, device, evaluator, config_dict, split = 'val', K_list = [10,20,50,100]):
    model.eval()

    model.refresh_all_embeddings(
            config_dict['num_users'],
            config_dict['num_items'],
            config_dict['user'],
            config_dict['item'],
            config_dict['item_attr_pair_dict'],
    )

    eval_user_idx_list = []
    recall_list_dict = {K: [] for K in K_list}

    test_batch_size = 100
    # iterate over all users
    for i in tqdm(range(0, evaluator.num_users_dict[split], test_batch_size)):
        user_idx = torch.arange(i, min(i + test_batch_size, evaluator.num_users_dict[split])).to(device)

        scores_pred = model.get_users_scores(user_idx)

        eval_users_idx, recall_dict_minibatch = evaluator.eval(scores_pred, user_idx, split = split, K_list = K_list)
        eval_user_idx_list.append(eval_users_idx)

        for K in K_list:
            recall_list_dict[K].append(recall_dict_minibatch[K])
    
    recall_dict = {K: np.average(np.concatenate(recall_list_dict[K])) for K in K_list}

    return recall_dict, model.x_user, model.x_item


def fix_m0(args):
    print('train m0 and fix it till the end')
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = DynRecDataset(name = args.dataset)
    pinsage_hyperparam_list = get_pinsage_hyperparam_list(dataset_name = args.dataset)
    
    ##### Prepare for training
    # G0: Training
    # G1: Validation
    edge_index_useritem_dict, num_users_dict, num_items_dict, _\
        = split_dynrecdataset(dataset, time_train = sliced_time_list[0], time_val = sliced_time_list[1])
    
    time_dict = {}
    split_list = ['train', 'val']
    for i, split in enumerate(split_list):
        time_dict[split] = sliced_time_list[i]
    
    print('====Basic stats')
    for i, split in enumerate(split_list):
        print(f'time: {sliced_time_list[i]}')
        print(f'#{split} users: ', num_users_dict[split])
        print(f'#{split} items: ', num_items_dict[split])
        print(f'#{split} edges: ', len(edge_index_useritem_dict[split][0]))
        print()

    item_attr_pair_dict = dataset.item_attr_pair_dict(time_dict['val'])
    for item_attr_name, (item_attr, item_attr_offset) in item_attr_pair_dict.items():
        item_attr_pair_dict[item_attr_name] = (item_attr.to(device), item_attr_offset.to(device))    

    train_dataset = RecDatasetNegsamling(
        edge_index_useritem_dict['train'][0], 
        edge_index_useritem_dict['train'][1], 
        num_users_dict['train'], 
        num_items_dict['train']
    )

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    train_config_dict = {
        'num_users': num_users_dict['train'],
        'num_items': num_items_dict['train'],
        'user': edge_index_useritem_dict['train'][0].to(device),
        'item': edge_index_useritem_dict['train'][1].to(device),
        'item_attr_pair_dict': item_attr_pair_dict,
    }

    ##### Prepare for evaluation
    eval_dict_list = []
    for i in range(len(sliced_time_list)-1):
        eval_dict = {}

        # Evaluate on the Gi+1\Gi graph
        edge_index_useritem_dict, num_users_dict, num_items_dict, _\
            = split_dynrecdataset(dataset, time_train = sliced_time_list[i], time_val = sliced_time_list[i+1])

        time_dict = {}
        time_dict['train'] = sliced_time_list[i]
        time_dict['val'] = sliced_time_list[i+1]

        for split in split_list:
            time = time_dict[split]
            print(f'time: {time}')
            print(f'#{split} users: ', num_users_dict[split])
            print(f'#{split} items: ', num_items_dict[split])
            print(f'#{split} edges: ', len(edge_index_useritem_dict[split][0]))
            print()
        
        item_attr_pair_dict = dataset.item_attr_pair_dict(time_dict['val'])
        for item_attr_name, (item_attr, item_attr_offset) in item_attr_pair_dict.items():
            item_attr_pair_dict[item_attr_name] = (item_attr.to(device), item_attr_offset.to(device)) 

        # Inference on Gi
        config_dict = {
            'num_users': num_users_dict['train'],
            'num_items': num_items_dict['train'],
            'user': edge_index_useritem_dict['train'][0].to(device),
            'item': edge_index_useritem_dict['train'][1].to(device),
            'item_attr_pair_dict': item_attr_pair_dict,
        }
        
        evaluator = RecallEvaluator(edge_index_useritem_dict, num_users_dict, num_items_dict)

        eval_dict['evaluator'] = evaluator
        eval_dict['config_dict'] = config_dict
        eval_dict['time_train'] = sliced_time_list[i]
        eval_dict['time_val'] = sliced_time_list[i+1]
        eval_dict_list.append(eval_dict)

    print('========PinSAGE hyperparameters')
    print(pinsage_hyperparam_list[0]['emb_dim'])
    print(pinsage_hyperparam_list[0]['num_layers'])
    print()

    ##### Prepare model and optimizer
    model = PinSAGE(
        emb_dim=pinsage_hyperparam_list[0]['emb_dim'],
        num_layers=pinsage_hyperparam_list[0]['num_layers'],
        item_encoder=ItemEncoder(pinsage_hyperparam_list[0]['emb_dim'], dataset.num_item_attrs_dict),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        if os.path.exists(args.log_dir):
            print('Removing existing tensorboard log..')
            shutil.rmtree(args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir)

    if args.checkpoint_path != '':
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok = True)

    K_list = [10, 20, 50, 100] # for recall @ K
    K_primary = 50 # use this for early stopping

    best_val_recall_dict_list = [{K: 0 for K in K_list} for i in range(len(eval_dict_list))]
    best_model = None
    best_embedding_list = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_dataset.sample_neg_batch() # sample a new set of negative items in each training epoch
        train_loss = train(model, device, train_loader, optimizer, args.weight_decay, train_config_dict)

        print(train_loss)

        if args.log_dir != '':
            writer.add_scalar('train/loss', train_loss, epoch)

        if epoch % args.log_every == 0:
            print('Evaluating...')
            
            val_recall_dict_list = []

            embedding_list = []

            train_recall_dict, emb_user, emb_item = eval(
                    model, 
                    device, 
                    eval_dict_list[0]['evaluator'], 
                    eval_dict_list[0]['config_dict'], 
                    split = 'train', 
                    K_list = K_list
            )

            print('Training')
            print(train_recall_dict)

            if args.log_dir != '':
                writer.add_scalar(f'train/{sliced_time_list[0]}/recall@{K_primary}', train_recall_dict[K_primary], epoch)

            for eval_dict in eval_dict_list:
                val_recall_dict, emb_user, emb_item = eval(
                    model, 
                    device, 
                    eval_dict['evaluator'], 
                    eval_dict['config_dict'], 
                    split = 'val', 
                    K_list = K_list
                )

                embedding_list.append((emb_user, emb_item))

                time_train = eval_dict['time_train']
                time_val = eval_dict['time_val']

                print(f'Val recall {time_val} minus {time_train}: {val_recall_dict}')

                if args.log_dir != '':
                    writer.add_scalar(f'val/{time_val}_minus_{time_train}/recall@{K_primary}', val_recall_dict[K_primary], epoch)

                val_recall_dict_list.append(val_recall_dict)
            
            if best_val_recall_dict_list[0][K_primary] < val_recall_dict_list[0][K_primary]:
                print('Achieved better val')
                best_val_recall_dict_list = val_recall_dict_list
                best_embedding_list = []
                for (emb_user, emb_item) in embedding_list:
                    best_embedding_list.append((emb_user.detach().clone(), emb_item.detach().clone()))
                best_model = copy.deepcopy(model)

    if args.checkpoint_path != '':
        print('Saving checkpoint...')
        checkpoint = {
            'best_val_recall_dict_list': best_val_recall_dict_list,
            'best_embedding_list': best_embedding_list,
            'model_state_dict': best_model.state_dict(),
            'time_list': sliced_time_list,
            'pinsage_hyperparam_list': pinsage_hyperparam_list,
            'args': args.__dict__}
        torch.save(checkpoint, args.checkpoint_path)

    if args.log_dir != '':
        writer.close()