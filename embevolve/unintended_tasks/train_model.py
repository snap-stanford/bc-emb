import argparse
import os
import pickle
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from recsys.shared import sliced_time_list
from recsys.dataset import DynRecDataset
from recsys.models import MLP
from recsys.utils import split_dynrecdataset, binarize, binarize_between, UserItemEdgeLevelDataset, NodeLevelDataset
from recsys.evaluate import UnintendedTaskEvaluator

def train(model, device, loader, loss_fun, optimizer):
    model.train()
    
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        y_pred = model(x.to(device))

        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        loss = loss_fun(y_pred, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()

    loss_accum = 0
    y_true_list = []
    y_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x_mini, y_true_mini = batch

        with torch.no_grad():
            y_pred_mini = model(x_mini.to(device)).cpu()

        if y_pred_mini.shape[1] == 1:
            y_pred_mini = y_pred_mini.flatten()

        y_true_list.append(y_true_mini)
        y_pred_list.append(y_pred_mini)

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)

    return evaluator.eval(y_pred, y_true)

def train_once(args, time_list, embedding_list, writer):
    print(f'train the prediction head once')
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and args.device > 0 else torch.device("cpu")
    dataset = DynRecDataset(name = args.dataset)

    init_edge_index_useritem_dict, init_num_users_dict, init_num_items_dict, init_additional_dict_dict\
        = split_dynrecdataset(dataset, time_list[0], time_list[1])

    min_value = None
    max_value = None

    if args.prediction_task == "binary-edge-rating":
        '''
            Binarized item rating prediction task
            - Task type: binary classification
            - Input: edges
            - Output: binarized edge rating (threshold is 4)
        '''
        thresh = 4
        train_dataset = UserItemEdgeLevelDataset(init_edge_index_useritem_dict['train'], 
                                                binarize(init_additional_dict_dict["train"]["edge_ratings"], thresh),
                                                user_emb = embedding_list[0][0],
                                                item_emb = embedding_list[0][1],)

        ind = 0
        edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
            = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
        val_dataset = UserItemEdgeLevelDataset(edge_index_useritem_dict['val'], 
                                    binarize(additional_dict_dict["val"]["edge_ratings"], thresh),
                                    user_emb = embedding_list[ind][0],
                                    item_emb = embedding_list[ind][1],)
        
        earlystop_val_dataset_idx = 0
        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1

    elif args.prediction_task == "user-activity":
        '''
            User activity prediction task
            - Task type: binary classification
            - Input: user nodes
            - Output: if a given user is active or not.
        '''
        train_dataset = NodeLevelDataset(init_additional_dict_dict["val"]["user_activity"],
                                        node_emb = embedding_list[0][0])

        ind = 1
        edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
            = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
        val_dataset = NodeLevelDataset(additional_dict_dict["val"]["user_activity"],
                                    node_emb = embedding_list[ind][0])

        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1

    elif args.prediction_task == "user-positive-activity":
        '''
            User positive activity prediction task
            - Task type: binary classification
            - Input: user nodes
            - Output: if a given user is active or not with a positive review.
                      Threshold for positive review is 4.
                      (Positive ratio is 29%)
        '''
        thresh = 4
        train_dataset = NodeLevelDataset(binarize(init_additional_dict_dict["val"]["user_activity_max_ratings"], thresh),
                                        node_emb = embedding_list[0][0])

        ind = 1
        edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
            = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
        val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["user_activity_max_ratings"], thresh),
                                    node_emb = embedding_list[ind][0])
        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1
    
    elif args.prediction_task == "binary-item-rating-std":
        '''
            Binarized item rating std prediction task
            - Task type: binary classification
            - Input: item nodes
            - Output: binarized item rating std (threshold is median)
            (Positive ratio is 50% because the threshold is median)
            Min is 0 and max is around 2, but only 100 items have std more than 1.5. Graph is exponential.
            Highly polarizing definition should probably be more polarizing than median.
        '''
        thresh = 1 # np.median(init_additional_dict_dict["train"]["std_item_ratings"])

        train_dataset = NodeLevelDataset(binarize(init_additional_dict_dict["train"]["std_item_ratings"], thresh),
                                        node_emb = embedding_list[0][1][init_additional_dict_dict["train"]["unique_items_for_avg_item_ratings"]])
        l = init_additional_dict_dict["train"]["std_item_ratings"]
        
        ind = 0        
        edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
            = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
        val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["std_item_ratings"], thresh),
                                    node_emb = embedding_list[ind][1][additional_dict_dict["val"]["unique_items_for_avg_item_ratings"]])
        
        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1

    elif args.prediction_task == "binary-item-rating-avg":
        '''
            Binarized item rating prediction task
            - Task type: binary classification
            - Input: item nodes
            - Output: binarized item rating (threshold is median)
        '''
        thresh = np.median(init_additional_dict_dict["train"]["avg_item_ratings"])

        train_dataset = NodeLevelDataset(binarize(init_additional_dict_dict["train"]["avg_item_ratings"], thresh),
                                        node_emb = embedding_list[0][1][init_additional_dict_dict["train"]["unique_items_for_avg_item_ratings"]])

        ind = 0        
        edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
            = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
        val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["avg_item_ratings"], thresh),
                                    node_emb = embedding_list[ind][1][additional_dict_dict["val"]["unique_items_for_avg_item_ratings"]])

        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1

    else:
        raise ValueError(f'Unknown downstream task called {args.prediction_task}.')

    args.metric = evaluator.metric

    print(f'=====Summary of {args.prediction_task} task')
    print(f'===#Training data: {len(train_dataset)}')
    print(f'===#Validation data: {len(val_dataset)}')
    print()

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    model = MLP(input_emb_dim = train_dataset.input_emb_dim,
                hidden_dim = args.hidden_dim,
                dropout = args.dropout,
                out_dim = out_dim,
                min_value = min_value,
                max_value = max_value).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    args.better = evaluator.better
    if evaluator.better == 'higher':
        best_val_perf = 0
    else:
        best_val_perf = 1000

    checkpoint = {}

    best_epoch = None
    best_model_state_dict = None

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        train_loss = train(model, device, train_loader, loss_fun, optimizer)
        print("Loss is", train_loss)

        if epoch % args.log_every == 0:
            val_perf = eval(model, device, val_loader, evaluator)
            
            if evaluator.better == 'higher':
                if best_val_perf < val_perf:
                    best_val_perf = val_perf
                    best_epoch = epoch
                    best_model_state_dict = copy.deepcopy(model.state_dict())
            else:
                if best_val_perf > val_perf:
                    best_val_perf = val_perf
                    best_epoch = epoch
                    best_model_state_dict = copy.deepcopy(model.state_dict())

            if writer is not None:
                writer.add_scalar(f'val/dropout{args.dropout}-dim{args.hidden_dim}/{evaluator.metric}', val_perf, epoch)
            
            print(f'dropout{args.dropout}-dim{args.hidden_dim}, val {evaluator.metric}: {val_perf}')


    checkpoint['best_val_perf'] = best_val_perf
    checkpoint['best_epoch'] = best_epoch
    checkpoint['model_state_dict'] = best_model_state_dict
    checkpoint['args'] = copy.deepcopy(args.__dict__)

    return checkpoint

def main():
    print("Starting run task.")

    # Training settings
    parser = argparse.ArgumentParser(description='Downstream tasks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default='Amazon-Video_Games',
                        help='dataset name (default: Amazon-Video_Games)')                  
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden embedding dimensionality (default: 128)')
    parser.add_argument('--prediction_task', type=str, default='edge-rating',
                        help='prediction target for the downstream task (default: edge-rating)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 2048)')                      
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--log_every', type=int, default=1,
                        help='how often we wanna evaluate the performance (default: 1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')   
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--model_path', type=str, default="",
                        help='model path')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of different seeds')
    args = parser.parse_args()

    if ('edge-rating' in args.prediction_task) and (args.dataset == 'Amazon-Grocery_and_Gourmet_Food'):
        args.epochs = 100

    print(args)

    torch.set_num_threads(1)

    if os.path.exists(args.model_path):
        print('Already found the saved checkpoint. Skipping...')
        exit(-1)

    if os.path.exists(args.log_dir):
        print('Removing existing tensorboard log..')
        shutil.rmtree(args.log_dir)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    args.embeddings_path = f'../intended_task/checkpoint/code-release/{args.dataset}/fix-m0'

    embeddings_chckpt = torch.load(args.embeddings_path, map_location="cpu")
    embedding_list = embeddings_chckpt['best_embedding_list']

    hidden_dim_list = [128, 256, 512, 1024]
    dropout_list =  [0, 0.25, 0.5]
    checkpoint_list = []
    val_perf_list = []

    for dropout in dropout_list:
        for hidden_dim in hidden_dim_list:
            args.dropout = dropout
            args.hidden_dim = hidden_dim
            checkpoint = train_once(args, sliced_time_list, embedding_list, writer)
            checkpoint_list.append(checkpoint)
            val_perf_list.append(checkpoint['best_val_perf'])

    val_perf_list = np.array(val_perf_list)
    print(val_perf_list)
    if args.better == 'higher':
        best_idx = np.argmax(val_perf_list)
    else:
        best_idx = np.argmin(val_perf_list)
    
    best_checkpoint = checkpoint_list[best_idx]
    print(f'Best val performance {val_perf_list[best_idx]}')

    if writer is not None:
        writer.close()

    best_checkpoint_list = []
    args.dropout = best_checkpoint['args']['dropout']
    args.hidden_dim = best_checkpoint['args']['hidden_dim']

    for seed in range(args.num_seeds):
        args.seed = seed
        best_checkpoint = train_once(args, sliced_time_list, embedding_list, writer = None)
        best_checkpoint_list.append(best_checkpoint)

    if args.model_path != '':
        os.makedirs(os.path.dirname(args.model_path), exist_ok = True)
        torch.save(best_checkpoint_list, args.model_path)
    

if __name__ == "__main__":
    main()