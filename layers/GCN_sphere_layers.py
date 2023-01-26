from torch_geometric.nn import GCNConv
import torch
from torch_sparse import SparseTensor
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCNConv_ACM(GCNConv):
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, w_for_norm=None, space_dims=None,
                names=None, layer_index=None, layer_bn_for_hyperplan=None,
                skip_weight=None, num_layers=None, args=None) -> Tensor:
        wm_fix = args.wm_fix
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if layer_index != 0 and layer_index <= num_layers - 2:
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            p0, a = p0_generate(w_for_norm, self.out_channels)
            b_for_hyperplan_mapping_value = 0.9 * a
            Q_p0_w = push_forward(x, p0, a, b=b_for_hyperplan_mapping_value)
            x_tmp = self.lin(Q_p0_w)
            x_tmp = layer_bn_for_hyperplan(x_tmp)
            x_tmp = torch.tanh(x_tmp)
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            x = push_back(x_tmp, p0, w_for_norm)


        elif layer_index == 0:
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            p0, a = p0_generate(w_for_norm, self.out_channels)
            x_tmp = self.lin(x)
            x_tmp = layer_bn_for_hyperplan(x_tmp)
            x_tmp = torch.tanh(x_tmp)
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            x = push_back(x_tmp, p0, w_for_norm)
        else:
            x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out = out + self.bias
        if layer_index == num_layers - 1:
            out = out
        else:
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            out = RiemannAgg(out, w_for_norm)

        return out

def RiemannAgg(x, w):
    squar_x = torch.square(x)
    squar_x_w = torch.mul(squar_x, w)
    sum_squar_x_w = torch.sum(squar_x_w, dim=1)
    sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-6)
    sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
    x = torch.div(x, sqrt_x_w)
    return x


def push_back(x, p0, w_for_norm):
    f_p0_v_Numerator = -2 * (x - p0) * w_for_norm @ p0.t()
    f_p0_v_denominator = (x - p0) * w_for_norm @ (x - p0).t()
    f_p0_v_denominator = torch.diag(f_p0_v_denominator).unsqueeze(dim=1)
    f_p0_v = f_p0_v_Numerator / f_p0_v_denominator
    x_tmp = f_p0_v * (x - p0) + p0
    return x_tmp


def p0_generate(w_for_norm, out_channels):
    p0 = torch.zeros((1, out_channels)).cuda()
    a = 1 / torch.sqrt(w_for_norm[0][0] + 1e-6)
    p0[0][0] = a.item()
    return p0, a


def push_forward(x, p0, a, b=0):
    x_tmp = x
    w1 = x[:, 0]
    g_po_w = (b - a) / (w1 - a)
    g_po_w = g_po_w.unsqueeze(dim=1)
    Q_p0_w = g_po_w * (x_tmp - p0) + p0
    return Q_p0_w
