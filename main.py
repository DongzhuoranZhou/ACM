from options.base_options import BaseOptions, reset_weight
from trainer import trainer_with_tensorboard
import torch
import os
import numpy as np
import random
import logging


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


seeds = [100, 200, 300, 400, 500]
layers_GCN = [2, 15, 30, 45, 60]
layers_SGCN = [5, 60, 120]


def main(args):
    acc_test_layers = []
    if args.type_model in ['GCN', 'GAT']:
        layers = layers_GCN
    else:
        layers = layers_SGCN
    if args.type_norm == 'group':
        args = reset_weight(args)
    for layer in layers:
        args.num_layers = layer
        acc_test_seeds = []
        for seed in seeds:
            args.random_seed = seed
            set_seed(args)
            logging.info("-" * 50)
            trnr = trainer_with_tensorboard(args)
            acc_test, acc_valid, loss_val, loss_test = trnr.train_compute_MI()
            acc_test_seeds.append(acc_test)
        avg_acc_test = np.mean(acc_test_seeds)
        acc_test_layers.append(avg_acc_test)
    print(
        f'experiment results of {args.type_layer} applied in {args.type_model} with layers {seeds} on dataset {args.dataset}')
    print('test accuracies: ', acc_test_layers)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
