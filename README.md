# T3gnn
In this repo we present T3GNN (GNN for temporal transaction and textual content) to perform future transaction prediction on Web3 ecosystems.

## Description
In Web3 social platforms, i.e. social web applications which rely on blockchain technology to support their functionalities, interactions among users are usually multimodal, from common social interactions such as following, liking, or posting, to specific relations given by crypto-token transfers facilitated by the blockchain. In this dynamic and intertwined networked context, our main goals are i) to predict whether a pair of users will be involved in a token transaction, i.e. the transaction prediction task, and ii) to verify whether performances may be enhanced by textual content. To address the above issues, we developed T3GNN, a solution that combines temporal graph networks, for handling the dynamic structure of financial transactions; sentence embeddings, for representing content; and historical negative sampling. We evaluate our approach in a Web3 context by leveraging a novel high-resolution temporal dataset, derived from one of the most used Web3 social platforms, which spans more than one year of financial interactions as well as textual published content. Results show that T3GNN yields prediction performances aligned to similar link prediction tasks trained using historical negative links. Moreover, the importance of textual features and sentence embeddings is still unclear, since the performance gain is not constant throughout the evaluation period, requesting further investigation.

## Architecture overview

The figure below shows the running pipeline of T3GNN.
![GNN Architecture](t3gnn-pipeline.png "T3GNN").
