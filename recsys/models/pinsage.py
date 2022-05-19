"""
Mostly adopted from https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py
"""
from typing import Dict, Optional
import math

import torch
from torch import nn
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F

import numpy as np
from scipy.sparse import csr_matrix

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def get_users_scores(self, users):
        raise NotImplementedError
    
class PinSAGE(BasicModel):
    def __init__(self, 
                 emb_dim: int, 
                 num_layers: int, 
                 item_encoder: torch.nn.Module,
                 num_users: Optional[int] = None,
                 ):
        '''
        PinSAGE assumes all items are associated with node features encoded by item_encoder
            Args:
            - emb_dim
                Embedding dimensionality
            - num_layers: 
                Number of GNN message passing layers
            - item_encoder: 
                torch.nn.Molecule that takes the raw input features as input and output the emb_dim embeddings of items
                item_encoder(item_feat) should give item_embeddings
            - num_users:
                None: The model is inductive for users
                Otherwise: the model learns unique shallow embedding for each user. the model is not inductive for users
        '''
        super(PinSAGE, self).__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.item_encoder = item_encoder
        self.f = nn.Sigmoid()

        self.x_user = None
        self.x_item = None

        self.num_users = num_users

        self.__init_weight()
        
    def __init_weight(self):
        # Uniform learnable user embedding

        if self.num_users is None:
            self.embedding_user = torch.nn.Embedding(
                num_embeddings=1, embedding_dim=self.emb_dim)
            torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        else:
            self.embedding_user = torch.nn.Embedding(
                num_embeddings=self.num_users, embedding_dim=self.emb_dim)
            torch.nn.init.normal_(self.embedding_user.weight, std=0.1)

        # SAGE
        self.sage_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.sage_convs.append(SAGEConv(self.emb_dim, self.emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

    def refresh_all_embeddings(self, 
                            num_users: int, 
                            num_items: int,
                            user: torch.LongTensor, 
                            item: torch.LongTensor, 
                            *args):
        '''
            Use the current GNN parameter and compute all the user/item embeddings.
            The embeddings are stored in x_user and x_item.

            Args: 
            - num_users
            - num_items

            - user
            - item
                user[i] interacts with item[i]
            - *args
                for item attr
        '''

        edge_index1 = torch.stack([user, item + num_users])
        edge_index2 = torch.stack([item + num_users, user])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        num_nodes = num_users + num_items
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

        # Prepare input feature
        if self.num_users is None:
            x_user = self.embedding_user(torch.zeros(num_users).to(torch.long).to(edge_index.device))
        else:
            assert num_users == self.num_users, 'num_users do not coincide'
            x_user = self.embedding_user.weight

        x_item = self.item_encoder(*args)

        x = torch.cat([x_user, x_item])

        # perform message passing
        for i in range(self.num_layers):
            x = self.sage_convs[i](x, adj.t())
            x = self.batch_norms[i](x)

            # no relu for the last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        self.x_user, self.x_item = torch.split(x, [num_users, num_items])
        
    def get_users_scores(self, users):
        users = users.long()
        users_emb = self.x_user[users]
        items_emb = self.x_item
        scores = self.f(torch.matmul(users_emb, items_emb.t()))
        return scores
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.x_user[users.long()]
        pos_emb   = self.x_item[pos.long()]
        neg_emb   = self.x_item[neg.long()]
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.x_user[users]
        items_emb = self.x_item[items]
        scores = self.f(torch.sum(users_emb*items_emb, dim=1))
        return scores


class ItemEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int, num_item_attrs_dict: Dict[str, int]):
        super(ItemEncoder, self).__init__()

        self.embedding_dict = torch.nn.ModuleDict()
        for item_attr_name, num_item_attrs in num_item_attrs_dict.items():
            embedding_bag = torch.nn.EmbeddingBag(num_item_attrs, emb_dim)
            self.embedding_dict.update({item_attr_name: embedding_bag})
    
    def forward(self, item_attr_pair_dict):
        emb = 0
        for item_attr_name, (item_attr, item_attr_offset) in item_attr_pair_dict.items():
            encoder = self.embedding_dict[item_attr_name]
            emb += encoder(item_attr, item_attr_offset)

        return emb