import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from recsys.models import PinSAGE, ItemEncoder
from recsys.utils import RecDatasetNegsamling, split_dynrecdataset, take_edge_index_diff
from recsys.evaluate import RecallEvaluator
from recsys.dataset import DynRecDataset
from recsys.shared import sliced_time_list, get_pinsage_hyperparam_list

import os
from tqdm import tqdm
import numpy as np
import random
import copy
import shutil

def train_trans(trans_fun, device, train_loader, optimizer_trans, old_embedding, new_embedding, args):
    loss_fun = torch.nn.MSELoss()
    
    trans_fun.train()

    emb_user_old, emb_item_old = old_embedding
    emb_user_new, emb_item_new = new_embedding

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        user, item, item_neg = batch
        user, item, item_neg = user.to(device), item.to(device), item_neg.to(device)

        loss_penalty = loss_fun(emb_user_old[user], trans_fun(emb_user_new[user]))+ \
                        loss_fun(emb_item_old[item], trans_fun(emb_item_new[item]))+ \
                            loss_fun(emb_item_old[item_neg], trans_fun(emb_item_new[item_neg]))

        optimizer_trans.zero_grad()
        loss_penalty.backward()
        optimizer_trans.step()

def train_eval_loop_trans(args, trans_fun, device, train_dataset, train_loader, optimizer_trans,\
                             old_embedding, new_embedding, time_train, time_val, writer):

    loss_fun = torch.nn.MSELoss()
    
    best_trans_fun = None
    best_loss = 1000

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_dataset.sample_neg_batch() # sample a new set of negative items in each training epoch
        train_trans(trans_fun, device, train_loader, optimizer_trans, old_embedding, new_embedding, args)

        if epoch % args.log_every == 0:
            print('Evaluating trans_fun...')

            # transform the embeddings into BC embeddings
            emb_user_new, emb_item_new = new_embedding
            emb_user_old, emb_item_old = old_embedding

            print('Transforming to BC embedddings')
            trans_fun.eval()
            emb_user_bc = trans_fun(emb_user_new).detach()
            emb_item_bc = trans_fun(emb_item_new).detach()

            user_emb_penalty = float(loss_fun(emb_user_old, emb_user_bc).detach().item())
            item_emb_penalty = float(loss_fun(emb_item_old, emb_item_bc).detach().item())

            print(f'User emb penalty: {user_emb_penalty}')
            print(f'Item emb penalty: {item_emb_penalty}')

            if args.log_dir != '':
                writer.add_scalar(f'emb_penalty/{time_val}_minus_{time_train}/user', user_emb_penalty, epoch)
                writer.add_scalar(f'emb_penalty/{time_val}_minus_{time_train}/item', item_emb_penalty, epoch)

            current_loss = 0.5 * (user_emb_penalty + item_emb_penalty)
            
            if (current_loss < best_loss) and (epoch > 50): # make sure the loss penalty is sufficiently decreased.
                print('Best val')
                best_loss = current_loss
                best_trans_fun = copy.deepcopy(trans_fun)

    return best_trans_fun

def posthoc_sloss(args):
    print(f'Posthoc training of Bk ({args.trans_type}) with single-step alignment loss.')
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = DynRecDataset(name = args.dataset)
    pinsage_hyperparam_list = get_pinsage_hyperparam_list(dataset_name = args.dataset)

    if args.log_dir != '':
        if os.path.exists(args.log_dir):
            print('Removing existing tensorboard log..')
            shutil.rmtree(args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    if args.checkpoint_path != '':
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok = True)

    checkpoint_vanilla = torch.load(args.checkpoint_vanilla_path)

    checkpoint = {
        'time_list': sliced_time_list,
        'args': args.__dict__,
        'pinsage_hyperparam_list': pinsage_hyperparam_list,
        'best_val_recall_dict_list': checkpoint_vanilla['best_val_recall_dict_list'],
        'best_embedding_list': [],
        'model_state_dict_list': checkpoint_vanilla['model_state_dict_list'],
    }

    print('Loading all vanilla models')
    model_list = []
    for i in range(len(sliced_time_list)-1):
        model = PinSAGE(
            emb_dim=pinsage_hyperparam_list[i]['emb_dim'],
            num_layers=pinsage_hyperparam_list[i]['num_layers'],
            item_encoder=ItemEncoder(pinsage_hyperparam_list[i]['emb_dim'], dataset.num_item_attrs_dict),
        )

        model.load_state_dict(checkpoint_vanilla['model_state_dict_list'][i])
        model.eval() # setting to eval mode

        model_list.append(model)
    
    print('Finished loading all models')

    best_trans_fun_list = []
    checkpoint['trans_fun_state_dict_list'] = []

    # train from scratch
    for i in range(len(sliced_time_list)-1):
        ##### Train on Gi and evaluate on the Gi+1\Gi

        time_train = sliced_time_list[i]
        time_val = sliced_time_list[i+1]

        print(f'=======Train on G{time_train}, evaluate on G{time_val}\G{time_train}')

        ##### Prepare for training
        edge_index_useritem_dict, num_users_dict, num_items_dict, _\
            = split_dynrecdataset(dataset, time_train = time_train, time_val = time_val)
        
        time_dict = {}        
        time_dict['train'] = time_train
        time_dict['val'] = time_val
        
        print('====Basic stats')
        split_list = ['train', 'val']
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

        train_dataset = RecDatasetNegsamling(
            edge_index_useritem_dict['train'][0], 
            edge_index_useritem_dict['train'][1], 
            num_users_dict['train'], 
            num_items_dict['train']
        )

        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

        config_dict = {
            'num_users': num_users_dict['train'],
            'num_items': num_items_dict['train'],
            'user': edge_index_useritem_dict['train'][0].to(device),
            'item': edge_index_useritem_dict['train'][1].to(device),
            'item_attr_pair_dict': item_attr_pair_dict,
        }

        ##### Prepare model and optimizer
        print('PinSAGE inference...')

        if i == 0:
            model = model_list[i]
            model.to(device)
            with torch.no_grad():
                model.refresh_all_embeddings(
                    config_dict['num_users'],
                    config_dict['num_items'],
                    config_dict['user'],
                    config_dict['item'],
                    config_dict['item_attr_pair_dict'],
                )
            
            best_embedding = (model.x_user.detach().cpu().clone(), model.x_item.detach().cpu().clone())

        else:
            new_model = model_list[i]
            old_model = model_list[i-1]

            new_model.to(device)
            with torch.no_grad():
                new_model.refresh_all_embeddings(
                    config_dict['num_users'],
                    config_dict['num_items'],
                    config_dict['user'],
                    config_dict['item'],
                    config_dict['item_attr_pair_dict'],
                )
                new_embedding = (new_model.x_user.detach(), \
                                    new_model.x_item.detach())

            old_model.to(device)
            with torch.no_grad():
                old_model.refresh_all_embeddings(
                    config_dict['num_users'],
                    config_dict['num_items'],
                    config_dict['user'],
                    config_dict['item'],
                    config_dict['item_attr_pair_dict'],
                )
                old_embedding = (old_model.x_user.detach(), \
                                    old_model.x_item.detach())
            
            if args.trans_type == 'linear':
                trans_fun = torch.nn.Linear(pinsage_hyperparam_list[i]['emb_dim'],\
                                            pinsage_hyperparam_list[i-1]['emb_dim'], bias = False).to(device)
            else:
                raise ValueError(f'Unknown transformation type called {args.trans_type}')

            optimizer_trans = optim.Adam(trans_fun.parameters(), lr=0.001)

            # learn transformation from new to old embeddings
            best_trans_fun = train_eval_loop_trans(args, trans_fun, device, train_dataset, train_loader, optimizer_trans, \
                                                    old_embedding, new_embedding, time_train, time_val, writer)
            best_trans_fun_list.append(best_trans_fun)
            checkpoint['trans_fun_state_dict_list'].append(best_trans_fun.state_dict())

            emb_user, emb_item = new_embedding

            # iteratively applying backward compatible transformation
            # the final embeddings are BC to the initial embeddings
            for best_trans_fun in best_trans_fun_list[::-1]:
                emb_user = best_trans_fun(emb_user)
                emb_item = best_trans_fun(emb_item)
            
            best_embedding = (emb_user.detach(), emb_item.detach())

            old_model.to('cpu')

        checkpoint['best_embedding_list'].append(best_embedding)

    if args.checkpoint_path != '':
        torch.save(checkpoint, args.checkpoint_path)
    
    if writer is not None:
        writer.close()
        
    



    