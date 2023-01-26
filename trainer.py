import torch
import os
from models.GCN import GCN
from models.simpleGCN import simpleGCN
from models.GAT import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
import scipy.sparse as sp
import torch_geometric.transforms as T
import torch.nn.functional as F
import glob
from torch_geometric.utils import remove_self_loops, add_self_loops
import numpy as np

def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, transform=T.NormalizeFeatures())[0]

        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data
    elif dataset in ['CoauthorCS']:
        data = Coauthor(path, 'cs', T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index

        # devide training validation and testing set
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_num = 40
        val_num = 150
        for i in range(15):  # number of labels
            index = (data.y == i).nonzero()[:, 0]
            perm = torch.randperm(index.size(0))
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num:(train_num + val_num)]]] = 1
            test_mask[index[perm[(train_num + val_num):]]] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')


def remove_feature(data, miss_rate):
    num_nodes = data.x.size(0)
    erasing_pool = torch.arange(num_nodes)[~data.train_mask]
    size = int(len(erasing_pool) * miss_rate)
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    x = data.x
    x[idx_erased] = 0.
    return x


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()



class trainer_with_tensorboard(object):
    def __init__(self, args):
        self.args=args
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        if self.dataset in ["Cora", "Citeseer", "Pubmed", 'CoauthorCS']:

            self.data = load_data(self.dataset)
            self.loss_fn = torch.nn.functional.nll_loss
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')

        self.miss_rate = args.miss_rate
        if self.miss_rate > 0.:
            self.data.x = remove_feature(self.data, self.miss_rate)
        self.type_model = args.type_model
        self.epochs = args.epochs
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay
        if self.type_model == 'GCN':
            self.model = GCN(args)
        elif self.type_model == 'simpleGCN':
            self.model = simpleGCN(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')

        self.data.to(self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.seed = args.random_seed
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight

    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            acc_train, acc_valid, acc_test, loss_val = self.run_testSet()
            return loss_train, acc_train, acc_valid, acc_test, loss_val
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e



    def train_compute_MI(self):
        best_acc = 0
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test, loss_val = self.train_net()
            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        # reload the best model
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # evaluate the saved model
        acc_valid, acc_test, loss_val, loss_test = self.run_testSet_After_Train()


        return acc_test, acc_valid, loss_val, loss_test,



    def run_trainSet(self):
        self.model.train()
        logits = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(logits[self.data.train_mask], 1)
        loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()


    def run_testSet_After_Train(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)

            logits_val = F.log_softmax(logits[self.data.val_mask], 1)
            loss_val = self.loss_fn(logits_val, self.data.y[self.data.val_mask]).item()
            logits_test = F.log_softmax(logits[self.data.test_mask], 1)
            loss_test = self.loss_fn(logits_test, self.data.y[self.data.test_mask]).item()
            logits = F.log_softmax(logits, 1)

        acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        return acc_valid, acc_test, loss_val, loss_test

    def run_testSet(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)

            logits_val = F.log_softmax(logits[self.data.val_mask], 1)
            loss_val = self.loss_fn(logits_val, self.data.y[self.data.val_mask]).item()
        logits = F.log_softmax(logits, 1)
        acc_train = evaluate(logits, self.data.y, self.data.train_mask)
        acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        return acc_train, acc_valid, acc_test, loss_val

    def filename(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        num_layers = int(self.model.num_layers)
        type_norm = self.type_norm
        miss_rate = int(self.miss_rate * 10)
        seed = int(self.seed)

        if type_norm == 'group':
            group = self.model.num_groups
            skip_weight = int(self.model.skip_weight * 1e3)

            filename = f'{filetype}_{type_model}_{type_norm}' \
                       f'L{num_layers}M{miss_rate}S{seed}G{group}S{skip_weight}.pth.tar'
        else:

            filename = f'{filetype}_{type_model}_{type_norm}' \
                       f'L{num_layers}M{miss_rate}S{seed}.pth.tar'

        filename = os.path.join(filedir, filename)
        return filename

    def get_saved_info(self, path=None):
        paths = glob.glob(path)
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 2)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 2, 'epoch')
        epochs.sort()
        return epochs

    def load_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        if os.path.exists(filename):
            print('load model: ', type_model, filename)
            return torch.load(filename)
        else:
            return None

    def save_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        state = self.model.state_dict()
        torch.save(state, filename)

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx

def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)