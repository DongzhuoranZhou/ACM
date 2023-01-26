from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.common_blocks import batch_norm
from layers import GAT_sphere_layers
import torch
import numpy as np


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_bn_for_hyperplan = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups
        self.args = args
        self.dropedge = args.dropedge
        self.cached = True if not args.dropedge else False
        # self.type_sphere_norm = args.type_sphere_norm
        self.type_layer = args.type_layer
        self.gat_layer = GATConv
        if self.type_layer in ['GATConv_ACM']:
            self.gat_hidden_layer = getattr(GAT_sphere_layers, self.type_layer)
            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)
        else:
            self.gat_hidden_layer = GATConv
        # build up the convolutional layers
        if self.num_layers == 1:
            self.layers_GCN.append(self.gat_layer(self.num_feats, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))
        elif self.num_layers == 2:
            self.layers_GCN.append(self.gat_hidden_layer(self.num_feats, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))
            self.layers_GCN.append(self.gat_layer(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))
        else:
            self.layers_GCN.append(self.gat_hidden_layer(self.num_feats, self.dim_hidden,heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))
            for _ in range(self.num_layers - 2):
                self.layers_GCN.append(self.gat_hidden_layer(self.dim_hidden, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))
            self.layers_GCN.append(self.gat_layer(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False,cached=self.cached))

        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))
        for i in range(self.num_layers):  # self.num_layers - 1
            dim_out = self.layers_GCN[i].out_channels
            skip_connect = True
            self.layers_bn_for_hyperplan.append(
                batch_norm(dim_out, 'batch', skip_connect, self.num_groups, self.skip_weight))


    def forward(self, x, edge_index):

        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.type_layer in ["GATConv_ACM"]:
                self.w_for_norm.data = self.w_for_norm.abs()
                if i <= self.num_layers - 2:
                    self.w_for_norm.data = self.w_for_norm.abs()
                    x = self.layers_GCN[i](x, edge_index, w_for_norm=self.w_for_norm,
                                           layer_index=i,
                                           layer_bn_for_hyperplan=self.layers_bn_for_hyperplan[i],
                                           num_layers=self.num_layers,
                                           args=self.args)

                    x = self.layers_bn[i](x)
                else:
                    self.w_for_norm.data = self.w_for_norm.abs()
                    if type(self.layers_GCN[i]) == GATConv:
                        x = self.layers_GCN[i](x, edge_index)  # for model with only 1 layer
                    else:
                        x = self.layers_GCN[i](x, edge_index, w_for_norm=self.w_for_norm,
                                               layer_index=i,
                                               layer_bn_for_hyperplan=self.layers_bn_for_hyperplan[i],
                                               num_layers=self.num_layers)
                    x = self.layers_bn[i](x)
            else:
                x = self.layers_GCN[i](x, edge_index)
                x = self.layers_bn[i](x)
                x = F.relu(x)

        return x