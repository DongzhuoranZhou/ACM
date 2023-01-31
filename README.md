# Alleviating Over-Smoothing via Aggregation over Compact Manifolds

This is an authors' implementation of "Alleviating Over-Smoothing via Aggregation over Compact Manifolds" in Pytorch.


## Requirements

python == 3.7

torch == 1.12.1

torch-geometric==2.2.0

torch_scatter==2.1.0

## Train over GCN, GAT or SGC backbone networks

To run our code, we can simply use the following command:
```
python main.py --cuda_num=0 --type_model=GCN --type_layer=GCNConv_ACM  --dataset=Cora --miss_rate=0.

python main.py --cuda_num=0 --type_model=GAT --type_layer=GATConv_ACM --dataset=Cora --miss_rate=0.

python main.py --cuda_num=0 --type_model=simpleGCN --type_layer=simpleGCN_ACM --dataset=Cora --miss_rate=0.
```

Hyperparameter explanations:

[//]: # (--type_norm: the type of normalization layer. We include ['None', 'batch', 'pair', 'group'] for none normalization, )

[//]: # (batch normalization, pair normalization and differentiable group normalization, respectively. )

--type_model: the type of GNN model. We include ['GCN', 'GAT', 'simpleGCN']

--type_layer: the type of our Method. We include ['GATConv_ACM', 'GCNConv_ACM', 'simpleGCN_ACM']

--dataset: we include ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']

--miss_rate: the missing rate of input features.
The value of 0. corresponds to the original dataset. The value of 1. means removing the features in validation and testing sets
