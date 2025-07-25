# T3GNN
In this repo, we present T3GNN (Temporal Graph Neural Network for Web3) to perform future transaction prediction on Web3 ecosystems.

## Description
In Web3 social platforms, i.e. social web applications that rely on blockchain technology to support their functionalities, interactions among users are usually multimodal, from common social interactions such as following, liking, or posting, to specific relations given by crypto-token transfers facilitated by the blockchain. In this dynamic and intertwined networked context, modeled as a financial network, our main goals are (i) to predict whether a pair of users will be involved in a financial transaction, i.e. the transaction prediction task, using textual information produced by users, and (ii) to verify whether performances may be enhanced by textual content. To address the above issues, we compared current snapshot-based temporal graph learning methods and developed T3GNN, a solution based on state-of-the-art temporal graph neural networks' design, which integrates fine-tuned sentence embeddings and a simple yet effective graph-augmentation strategy, for representing content, and historical negative sampling. We evaluated models in a Web3 context by leveraging a novel high-resolution temporal dataset, collected from one of the most used Web3 social platforms, which spans more than one year of financial interactions as well as published textual content. The experimental evaluation has shown that T3GNN consistently achieved the best performance over time and for most of the snapshots. Furthermore, through an extensive analysis of the performance of our model, we show that, despite the graph structure being crucial for making predictions, textual content contains useful information for forecasting transactions, highlighting an interplay between users' interests and economic relationships in Web3 platforms. Finally, the evaluation has also highlighted the importance of adopting sampling methods alternative to random negative sampling when dealing with prediction tasks on temporal networks.

## Architecture Overview

The figure below shows the running pipeline of T3GNN. You can find the implementation of T3GNN in the `t3gnn.py` src file.  

![T3GNN pipeline](t3gnn-pipeline.png "T3GNN pipeline").

## Run T3GNN on your dataset

To feed T3GNN, a snapshot-based or discrete-time temporal graph is a list of PyG Data. The i-th element in the list corresponds to the representation of the i-th snapshot of the temporal network. If you are not familiar with PyG Data structures, node feature matrix, and edge_index, you can refer to [PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html). To use T3GNN with a custom dataset, you can follow these steps:
1. Create a folder `dirname` that will contain the edge list and node features of your snapshot graphs.
2. Move into `dirname` the edge_index tensors of your snapshot graphs by adopting the following convention: the edge list of the i-th snapshot is called `i_edge_index.pt`. It is important to start counting from zero. 
3. Optionally, move into `dirname` the node feature matrix of your snapshot graphs by adopting the following convention: the node feature matrix of the i-th snapshot is called `i_x.pt`. It is important to start counting from zero.
4. Train T3GNN on your dataset by running:

    ```
    python run.py --dataset dirname
    ```
In case you do not have node features, you must specify the number of nodes on your dataset by running:
    ```
    python run.py --dataset dirname --num_nodes num_nodes
    ```
You can set the hidden dimension of your model, specify the fixed random seed and the addition of self-loops by running:
    ```
    python run.py --dataset dirname --add_self_loops --hidden_dim 64 --seed 42
    ```
    
## Steemit Data
Due to privacy reasons on personal data like username and textual content, we can't release the dataset related to Steemit. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are used to feed the models. For data gathering, you can refer to the [Steemit API](https://developers.steem.io/) documentation.

## Experiments
### Reproduce
To reproduce all the experiments presented in the paper, you can easily run the `T3GNN-Steemit.ipynb` notebook. Tu run it, you need first to install the requirements with:
```
pip install -r requirements.txt
```
### Additional information
We developed T3GNN using [Pytorch Geometric (PyG)](https://pyg.org/). For EvolveGCN and GCRN-GRU, we use the implementation available in [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/). We ran our experiments on NVIDIA Corporation GP107GL [Quadro P400]. In all our experiments, we use the Adam optimizer. This choice was made according to some prior works on GNN architecture for temporal networks (e.g. ROLAND, EvolveGCN). We adopt the live-update setting to train and evaluate the models, wherein we engage in incremental training and assess their performance across all the available snapshots. With respect to each snapshot, a random selection of 25\% of edges is employed to establish the early-stopping condition (validation set), while the remaining 75\% are utilized as the training set. The edges contained within the subsequent snapshot constitute the test set. Consistently, we apply identical dataset divisions and training procedures across all the methods. We report the configuration of hyperparameters for T3GNN in the following table: 

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam  |
| Learning rate  | 0.01  |
| Weight Decay   | 5e-3  |
| Epochs         | 50    | 

Overall, computing all the experiments with all the baselines and a single configuration of hyperparameters takes no more than 30 minutes.

### Cite
If you use the code of this repository for your project or you find the work interesting, please cite the following work:  

Dileo, M., Zignani, M. Discrete-time graph neural networks for transaction prediction in Web3 social platforms. Mach Learn (2024). https://doi.org/10.1007/s10994-024-06579-y


```bibtex
﻿@Article{Dileo2024discrete,
author={Dileo, Manuel
and Zignani, Matteo},
title={Discrete-time graph neural networks for transaction prediction in Web3 social platforms},
journal={Machine Learning},
year={2024},
month={Jun},
day={25},
abstract={In Web3 social platforms, i.e. social web applications that rely on blockchain technology to support their functionalities, interactions among users are usually multimodal, from common social interactions such as following, liking, or posting, to specific relations given by crypto-token transfers facilitated by the blockchain. In this dynamic and intertwined networked context, modeled as a financial network, our main goals are (i) to predict whether a pair of users will be involved in a financial transaction, i.e. the transaction prediction task, even using textual information produced by users, and (ii) to verify whether performances may be enhanced by textual content. To address the above issues, we compared current snapshot-based temporal graph learning methods and developed T3GNN, a solution based on state-of-the-art temporal graph neural networks' design, which integrates fine-tuned sentence embeddings and a simple yet effective graph-augmentation strategy for representing content, and historical negative sampling. We evaluated models in a Web3 context by leveraging a novel high-resolution temporal dataset, collected from one of the most used Web3 social platforms, which spans more than one year of financial interactions as well as published textual content. The experimental evaluation has shown that T3GNN consistently achieved the best performance over time and for most of the snapshots. Furthermore, through an extensive analysis of the performance of our model, we show that, despite the graph structure being crucial for making predictions, textual content contains useful information for forecasting transactions, highlighting an interplay between users' interests and economic relationships in Web3 platforms. Finally, the evaluation has also highlighted the importance of adopting sampling methods alternative to random negative sampling when dealing with prediction tasks on temporal networks.},
issn={1573-0565},
doi={10.1007/s10994-024-06579-y},
url={https://doi.org/10.1007/s10994-024-06579-y}
}
```
