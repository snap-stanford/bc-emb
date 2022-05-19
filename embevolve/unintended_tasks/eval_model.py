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

def infer(args, time_list, embedding_list, model_state_dict):
    print(f'train the prediction head at timestamp {time_list[0]} once on a downstream task')
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

        val_dataset_list = []
        for ind in range(0, len(time_list) - 1):
            edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
                = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
            val_dataset = UserItemEdgeLevelDataset(edge_index_useritem_dict['val'], 
                                        binarize(additional_dict_dict["val"]["edge_ratings"], thresh),
                                        user_emb = embedding_list[ind][0],
                                        item_emb = embedding_list[ind][1],)
            val_dataset_list.append(val_dataset)
        
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

        val_dataset_list = []
        for ind in range(0, len(time_list) - 1):
            edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
                = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
            val_dataset = NodeLevelDataset(additional_dict_dict["val"]["user_activity"],
                                        node_emb = embedding_list[ind][0])
            val_dataset_list.append(val_dataset)

        earlystop_val_dataset_idx = 1
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

        val_dataset_list = []
        for ind in range(0, len(time_list) - 1):
            edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
                = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
            val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["user_activity_max_ratings"], thresh),
                                        node_emb = embedding_list[ind][0])

            val_dataset_list.append(val_dataset)

        earlystop_val_dataset_idx = 1
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


        val_dataset_list = []
        for ind in range(0, len(time_list) - 1):
            edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
                = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
            val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["avg_item_ratings"], thresh),
                                        node_emb = embedding_list[ind][1][additional_dict_dict["val"]["unique_items_for_avg_item_ratings"]])
            val_dataset_list.append(val_dataset)
        
        earlystop_val_dataset_idx = 0
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
            Min in 0 and max is around 2, but only 100 items have std more than 1.5. Graph is exponential.
            Highly polarizing definition should probably be more polarizing than median.
        '''
        thresh = 1
        train_dataset = NodeLevelDataset(binarize(init_additional_dict_dict["train"]["std_item_ratings"], thresh),
                                        node_emb = embedding_list[0][1][init_additional_dict_dict["train"]["unique_items_for_avg_item_ratings"]])

        val_dataset_list = []
        for ind in range(0, len(time_list) - 1):
            edge_index_useritem_dict, num_users_dict, num_items_dict, additional_dict_dict\
                = split_dynrecdataset(dataset, time_list[ind], time_list[ind+1])
            val_dataset = NodeLevelDataset(binarize(additional_dict_dict["val"]["std_item_ratings"], thresh),
                                    node_emb = embedding_list[ind][1][additional_dict_dict["val"]["unique_items_for_avg_item_ratings"]])
            val_dataset_list.append(val_dataset)
        
        earlystop_val_dataset_idx = 0
        loss_fun = torch.nn.BCEWithLogitsLoss()
        evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
        out_dim = 1

    else:
        raise ValueError(f'Unknown downstream task called {args.prediction_task}.')

    args.metric = evaluator.metric
    args.earlystop_val_dataset_idx = earlystop_val_dataset_idx

    print(f'=====Summary of {args.prediction_task} task')
    val_dataset_len_list = np.array([len(val_dataset) for val_dataset in val_dataset_list])
    print(f'===Avg #validation data: {val_dataset_len_list.mean()}')
    print()

    val_loader_list = []
    for i in range(len(time_list) - 1):
        val_loader = DataLoader(val_dataset_list[i], batch_size = args.batch_size, shuffle = False)
        val_loader_list.append(val_loader)

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

    model.load_state_dict(model_state_dict)

    val_perf_list = []
    for i in range(len(time_list) - 1):
        val_perf = eval(model, device, val_loader_list[i], evaluator)
        val_perf_list.append(val_perf)

    return val_perf_list

def main():
    print("Starting inference.")

    # Training settings
    parser = argparse.ArgumentParser(description='Downstream tasks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default='Amazon-Video_Games',
                        help='dataset name (default: Amazon-Video_Games)')                    
    parser.add_argument('--prediction_task', type=str, default='binary-edge-rating',
                        help='prediction target for the downstream task (default: binary-edge-rating)')
    parser.add_argument('--embgen_strategy', type=str, default='fix-m0',
                        help='Embedding generation strategy (default: fix-m0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')   
    parser.add_argument('--model_path', type=str, default="",
                        help='model path')
    parser.add_argument('--result_path', type=str, default="",
                        help='result path')
    args = parser.parse_args()

    torch.set_num_threads(1)

    if not os.path.exists(args.model_path):
        raise RuntimeError(f'Model checkpoint not found at {args.model_path}')

    if os.path.exists(args.result_path):
        print('Results already saved.')
        exit(-1)

    args.embeddings_path = f'../intended_task/checkpoint/code-release/{args.dataset}/{args.embgen_strategy}'

    embeddings_chckpt = torch.load(args.embeddings_path, map_location="cpu")
    embedding_list = embeddings_chckpt['best_embedding_list']

    if args.embgen_strategy == 'nobc':
        emb_user, emb_item = embedding_list[0]
        init_dim = emb_user.shape[1]
        for i in range(len(embedding_list)):
            emb_user, emb_item = embedding_list[i]
            embedding_list[i] = (emb_user[:,:init_dim], emb_item[:,:init_dim])

    model_checkpoint_list = torch.load(args.model_path)
    val_perf_mat = []

    for model_checkpoint in model_checkpoint_list:
        args.__dict__.update(model_checkpoint['args'])
        print(args)
        
        val_perf_list = infer(args, sliced_time_list, embedding_list, model_checkpoint['model_state_dict'])

        print(val_perf_list)
        val_perf_mat.append(val_perf_list)
        
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok = True)
        torch.save(val_perf_mat, args.result_path)
    

if __name__ == "__main__":
    main()