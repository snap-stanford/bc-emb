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

def train_pinsage(model, device, loader, optimizer, weight_decay, config_dict):
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
def eval_pinsage(model, device, evaluator, config_dict, split = 'val', K_list = [10,20,50,100]):
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

def train_eval_loop_pinsage(args, model, device, train_dataset, train_loader, optimizer, train_config_dict,\
                            eval_dict, K_list, K_primary, time_train, time_val, writer):
    
    best_val_recall_dict = {K: 0 for K in K_list}
    best_embedding = None
    best_model = None

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_dataset.sample_neg_batch() # sample a new set of negative items in each training epoch
        train_loss = train_pinsage(model, device, train_loader, optimizer, args.weight_decay, train_config_dict)

        if epoch % args.log_every == 0:
            print('Evaluating...')

            val_recall_dict, emb_user, emb_item = eval_pinsage(
                model, 
                device, 
                eval_dict['evaluator'], 
                eval_dict['config_dict'], 
                split = 'val', 
                K_list = K_list
            )

            print(f'Val recall {time_val} minus {time_train}: {val_recall_dict}')

            if writer is not None:
                writer.add_scalar(f'val/{time_val}_minus_{time_train}/recall@{K_primary}', val_recall_dict[K_primary], epoch)

            if best_val_recall_dict[K_primary] < val_recall_dict[K_primary]:
                best_val_recall_dict = val_recall_dict
                best_embedding = (emb_user.detach().clone(), emb_item.detach().clone())
                best_model = copy.deepcopy(model)

    return best_embedding, best_val_recall_dict, best_model


def train_pinsage_penalize(model, device, train_loader, optimizer, train_config_dict, x_user_prev_model, x_item_prev_model, args, trans_fun, optimizer_trans):
    model.train()
    if trans_fun is not None:
        trans_fun.train()

    loss_accum = 0

    lam = args.lam
    loss_fun = torch.nn.MSELoss()
    
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        user, item, item_neg = batch
        user, item, item_neg = user.to(device).long(), item.to(device).long(), item_neg.to(device).long()

        model.refresh_all_embeddings(
            train_config_dict['num_users'],
            train_config_dict['num_items'],
            train_config_dict['user'],
            train_config_dict['item'],
            train_config_dict['item_attr_pair_dict'],
        )

        loss, reg_loss = model.bpr_loss(user, item, item_neg)
        
        if trans_fun is None:
            target_dim = optimizer_trans # only use the first target_dim
            loss_penalty = loss_fun(x_user_prev_model[user], model.x_user[user][:, :target_dim])+ \
                            loss_fun(x_item_prev_model[item], model.x_item[item][:, :target_dim])+ \
                             loss_fun(x_item_prev_model[item_neg], model.x_item[item_neg][:, :target_dim])
        else:
            loss_penalty = loss_fun(x_user_prev_model[user], trans_fun(model.x_user[user]))+ \
                            loss_fun(x_item_prev_model[item], trans_fun(model.x_item[item]))+ \
                             loss_fun(x_item_prev_model[item_neg], trans_fun(model.x_item[item_neg]))

        loss = loss + reg_loss * args.weight_decay + loss_penalty * args.lam

        optimizer.zero_grad()
        if trans_fun is not None:
            optimizer_trans.zero_grad()
        
        loss.backward()

        optimizer.step()
        if trans_fun is not None:
            optimizer_trans.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def train_eval_loop_pinsage_penalize(args, model, device, train_dataset, train_loader, optimizer, train_config_dict,\
                                eval_dict, K_list, K_primary, time_train, time_val, writer, x_user_prev_model, x_item_prev_model, trans_fun, optimizer_trans):
    
    loss_fun = torch.nn.MSELoss()
    
    best_val_recall_dict = {K: 0 for K in K_list}
    best_embedding = None
    best_model = None
    best_trans_fun = None

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_dataset.sample_neg_batch() # sample a new set of negative items in each training epoch
        train_loss = train_pinsage_penalize(model, device, train_loader, optimizer, train_config_dict, x_user_prev_model, x_item_prev_model, args, trans_fun, optimizer_trans)

        print(f'Trainining loss: {train_loss}')

        if epoch % args.log_every == 0:
            print('Evaluating...')        
            val_recall_dict, emb_user, emb_item = eval_pinsage(
                model, 
                device, 
                eval_dict['evaluator'], 
                eval_dict['config_dict'], 
                split = 'val', 
                K_list = K_list
            )

            # transform the embeddings into BC embeddings
            if trans_fun is None:
                target_dim = optimizer_trans
                print(f'Using the first {target_dim} dimensions')
                emb_user_bc = emb_user.detach()[:,:target_dim]
                emb_item_bc = emb_item.detach()[:,:target_dim]
            else:
                print('Transforming to BC embedddings')
                trans_fun.eval()
                emb_user_bc = trans_fun(emb_user).detach()
                emb_item_bc = trans_fun(emb_item).detach()

            print(f'Val recall {time_val} minus {time_train}: {val_recall_dict}')

            user_emb_penalty = float(loss_fun(x_user_prev_model, emb_user_bc).detach().item())
            item_emb_penalty = float(loss_fun(x_item_prev_model, emb_item_bc).detach().item())

            print(f'User emb penalty: {user_emb_penalty}')
            print(f'Item emb penalty: {item_emb_penalty}')

            if args.log_dir != '':
                writer.add_scalar(f'val/{time_val}_minus_{time_train}/recall@{K_primary}', val_recall_dict[K_primary], epoch)
                writer.add_scalar(f'emb_penalty/{time_val}_minus_{time_train}/user', user_emb_penalty, epoch)
                writer.add_scalar(f'emb_penalty/{time_val}_minus_{time_train}/item', item_emb_penalty, epoch)
            
            if (best_val_recall_dict[K_primary] < val_recall_dict[K_primary]) and (epoch > 50): # make sure the loss penalty is sufficiently decreased.
                best_val_recall_dict = val_recall_dict
                best_embedding = (copy.deepcopy(emb_user), copy.deepcopy(emb_item))
                best_model = copy.deepcopy(model)
                best_trans_fun = copy.deepcopy(trans_fun)

    return best_embedding, best_val_recall_dict, best_model, best_trans_fun


def joint_sloss(args):
    print(f'Joint training of Mk and Bk ({args.trans_type}) with single-step alignment loss.')
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
    
    K_list = [10, 20, 50, 100] # for recall @ K
    K_primary = 50 # use this for early stopping

    checkpoint = {
        'time_list': sliced_time_list,
        'args': args.__dict__,
        'pinsage_hyperparam_list': pinsage_hyperparam_list,
        'best_val_recall_dict_list': [],
        'best_embedding_list': [],
        'model_state_dict_list': [],
    }

    if args.trans_type == 'linear':
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

        train_config_dict = {
            'num_users': num_users_dict['train'],
            'num_items': num_items_dict['train'],
            'user': edge_index_useritem_dict['train'][0].to(device),
            'item': edge_index_useritem_dict['train'][1].to(device),
            'item_attr_pair_dict': item_attr_pair_dict,
        }

        eval_dict = {}
        
        evaluator = RecallEvaluator(edge_index_useritem_dict, num_users_dict, num_items_dict)

        eval_dict['evaluator'] = evaluator
        eval_dict['config_dict'] = train_config_dict
        eval_dict['time_train'] = time_train
        eval_dict['time_val'] = time_val

        ##### Prepare model and optimizer
        print('Training PinSAGE...')
        # train pinsage in the first timestamp
        model = PinSAGE(
            emb_dim=pinsage_hyperparam_list[i]['emb_dim'],
            num_layers=pinsage_hyperparam_list[i]['num_layers'],
            item_encoder=ItemEncoder(pinsage_hyperparam_list[i]['emb_dim'], dataset.num_item_attrs_dict),
        ).to(device)

        print('========PinSAGE hyperparameters')
        print(pinsage_hyperparam_list[i]['emb_dim'])
        print(pinsage_hyperparam_list[i]['num_layers'])
        print()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if i == 0:
            best_embedding, best_val_recall_dict, best_model =\
                train_eval_loop_pinsage(args, model, device, train_dataset, train_loader, optimizer, train_config_dict,\
                                eval_dict, K_list, K_primary, time_train, time_val, writer)

            prev_model = best_model
            prev_model.eval()

        else:
            if args.trans_type == 'linear':
                trans_fun = torch.nn.Linear(pinsage_hyperparam_list[i]['emb_dim'],\
                                            pinsage_hyperparam_list[i-1]['emb_dim'], bias = False).to(device)
                optimizer_trans = optim.Adam(trans_fun.parameters(), lr=0.001)
            elif args.trans_type == 'notrans':
                trans_fun = None
                optimizer_trans = pinsage_hyperparam_list[i-1]['emb_dim'] # stores the target dimensionality
            else:
                raise ValueError(f'Unknown transformation type called {args.trans_type}')

            prev_model.refresh_all_embeddings(
                train_config_dict['num_users'],
                train_config_dict['num_items'],
                train_config_dict['user'],
                train_config_dict['item'],
                train_config_dict['item_attr_pair_dict'],
            )

            x_user_prev_model, x_item_prev_model = prev_model.x_user.detach(), prev_model.x_item.detach()
            
            best_embedding, best_val_recall_dict, best_model, best_trans_fun=\
                train_eval_loop_pinsage_penalize(args, model, device, train_dataset, train_loader, optimizer, train_config_dict,\
                                eval_dict, K_list, K_primary, time_train, time_val, writer, x_user_prev_model, x_item_prev_model, trans_fun, optimizer_trans)

            emb_user, emb_item = best_embedding

            if best_trans_fun is None:
                # the first init_target_dim is BC to the very first GNN model
                init_target_dim = pinsage_hyperparam_list[0]['emb_dim']
                emb_user = emb_user[:,:init_target_dim]
                emb_item = emb_item[:,:init_target_dim]
            else:
                best_trans_fun_list.append(best_trans_fun)
                checkpoint['trans_fun_state_dict_list'].append(best_trans_fun.state_dict())
                # iteratively applying backward compatible transformation
                # the final embeddings are BC to the initial embeddings
                for best_trans_fun in best_trans_fun_list[::-1]:
                    emb_user = best_trans_fun(emb_user)
                    emb_item = best_trans_fun(emb_item)

            best_embedding = (emb_user.detach(), emb_item.detach())

            prev_model = best_model
            prev_model.eval()

        print(best_embedding)
        print(best_val_recall_dict)

        checkpoint['best_embedding_list'].append(best_embedding)
        checkpoint['best_val_recall_dict_list'].append(best_val_recall_dict)
        checkpoint['model_state_dict_list'].append(best_model.state_dict())

    if args.checkpoint_path != '':
        torch.save(checkpoint, args.checkpoint_path)
    
    if writer is not None:
        writer.close()
        
    



    