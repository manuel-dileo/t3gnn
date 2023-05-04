# T3GNN
In this repo we present T3GNN (GNN for temporal transaction and textual content) to perform future transaction prediction on Web3 ecosystems.

## Description
In Web3 social platforms, i.e. social web applications which rely on blockchain technology to support their functionalities, interactions among users are usually multimodal, from common social interactions such as following, liking, or posting, to specific relations given by crypto-token transfers facilitated by the blockchain. In this dynamic and intertwined networked context, our main goals are i) to predict whether a pair of users will be involved in a token transaction, i.e. the transaction prediction task, and ii) to verify whether performances may be enhanced by textual content. To address the above issues, we developed T3GNN, a solution that combines temporal graph networks, for handling the dynamic structure of financial transactions; sentence embeddings, for representing content; and historical negative sampling. We evaluate our approach in a Web3 context by leveraging a novel high-resolution temporal dataset, derived from one of the most used Web3 social platforms, which spans more than one year of financial interactions as well as textual published content. Results show that T3GNN yields prediction performances aligned to similar link prediction tasks trained using historical negative links. Moreover, the importance of textual features and sentence embeddings is still unclear, since the performance gain is not constant throughout the evaluation period, requesting further investigation.

## Architecture overview

The figure below shows the running pipeline of T3GNN. You can find the implementation of T3GNN in the `t3gnn.py` src file.  

![T3GNN pipeline](t3gnn-pipeline.png "T3GNN pipeline").

## Steemit Data
Due to privacy reasons on personal data like username and textual content, we can't release the dataset related to Steemit. To patch this problem, we will provide an anonymized version of our data. This version represents the final mathematical objects that are use to feed the models. For data gathering you can refer to the [Steemit API](https://developers.steem.io/) documentation. Steemit transaction and textual data will be available soon

## Experiments
For the experiments presented in "Temporal graph networks and sentence embedding for transaction prediction in dynamic Web3 platforms", you can refer to the `T3GNN-Steemit.ipynb` notebook
