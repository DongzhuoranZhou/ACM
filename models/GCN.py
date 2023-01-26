from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.common_blocks import batch_norm
import torch
import numpy as np
from layers import GCN_sphere_layers
from torch_geometric.utils import dropout_edge


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.dropout = args.dropout
        self.dropedge = args.dropedge
        self.cached = True if not args.dropedge else False

        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_bn_for_hyperplan = nn.ModuleList([])

        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.type_layer = args.type_layer
        self.gcn_layer = GCNConv
        self.args = args

        if self.type_layer in ["GCNConv_ACM"]:
            self.gcn_hidden_layer = getattr(GCN_sphere_layers, self.type_layer)
            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)

        else:

            self.gcn_hidden_layer = GCNConv

        if self.num_layers == 1:
            self.layers_GCN.append(self.gcn_layer(self.num_feats, self.num_classes, cached=self.cached, bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(
                self.gcn_hidden_layer(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(
                self.gcn_layer(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        else:
            self.layers_GCN.append(
                self.gcn_hidden_layer(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            for _ in range(self.num_layers - 2):
                self.layers_GCN.append(
                    self.gcn_hidden_layer(self.dim_hidden, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(
                self.gcn_layer(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            self.layers_bn.append(
                batch_norm(dim_out, self.type_norm, self.skip_weight))
        for i in range(self.num_layers - 1):
            dim_out = self.layers_GCN[i].out_channels
            self.layers_bn_for_hyperplan.append(
                batch_norm(dim_out, 'batch', self.skip_weight))


    def forward(self, x, edge_index):
        if self.dropedge:
            edge_index = dropout_edge(edge_index, p=1 - self.dropedge, training=self.training)[0]
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.type_layer in ["GCNConv_ACM"]:
                self.w_for_norm.data = self.w_for_norm.abs()
                if i <= self.num_layers - 2:
                    self.w_for_norm.data = self.w_for_norm.abs()
                    x = self.layers_GCN[i](x, edge_index, w_for_norm=self.w_for_norm,
                                            layer_index=i,
                                           layer_bn_for_hyperplan=self.layers_bn_for_hyperplan[i],
                                           num_layers=self.num_layers,
                                            args=self.args,)
                    x = self.layers_bn[i](x)
                else:
                    self.w_for_norm.data = self.w_for_norm.abs()
                    if type(self.layers_GCN[i]) == GCNConv:
                        x = self.layers_GCN[i](x, edge_index)  # for model with only 1 layer
                    else:
                        x = self.layers_GCN[i](x, edge_index=edge_index, w_for_norm=self.w_for_norm,
                                                layer_index=i,
                                               layer_bn_for_hyperplan=self.layers_bn_for_hyperplan[i],
                                               num_layers=self.num_layers,
                                               args=self.args)
                    x = self.layers_bn[i](x)
            else:
                x = self.layers_GCN[i](x, edge_index)
                x = self.layers_bn[i](x)
                x = F.relu(x)
        return x
