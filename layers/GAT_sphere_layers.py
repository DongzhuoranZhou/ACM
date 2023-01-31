from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn import GATConv


class GATConv_ACM(GATConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, w_for_norm=None, layer_bn_for_hyperplan=None,
                return_attention_weights=None, layer_index=None, num_layers=None,
                args=None):
        wm_fix = args.wm_fix
        H, C = self.heads, self.out_channels

        if layer_index == 0:
            w_for_norm.data = torch.ones_like(w_for_norm)
            p0, a = p0_generate(w_for_norm, self.out_channels)

            x = self.lin_src(x)

            x = torch.tanh(x)
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            x = push_back(x, p0, w_for_norm)
            x_src = x_dst = x.view(-1, H, C)

        elif layer_index != 0 and layer_index <= num_layers - 2:
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            p0, a = p0_generate(w_for_norm, self.out_channels)
            b = 0.9*a
            Q_p0_w = push_forward(x, p0, a, b=b)
            x = self.lin_src(Q_p0_w)

            x = torch.tanh(x)
            # TODO PB(\sigma(xw))
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            x = push_back(x, p0, w_for_norm)
            x_src = x_dst = x.view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)



        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias


        if layer_index == num_layers - 1:
            out = out
        else:
            if wm_fix:
                w_for_norm.data = torch.ones_like(w_for_norm)
            out = RiemannAgg(out, w_for_norm)


        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
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