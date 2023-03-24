# Disentangled Graph Neural Network
This repository is for DGNN model by <a href='https://github.com/EAJOY'>@EAJOY</a>, proposed in the following paper:

><a href='https://akaxlh.github.io/'>Lianghao Xia</a>+, Yizhen Shao+, <a href='https://sites.google.com/view/chaoh/'>Chao Huang</a>*, Yong Xu, Huance Xu and Jian Pei. <i>Disentangled Graph Social Recommendation</i>. <a href='https://arxiv.org/pdf/2303.07810.pdf'>Paper in arXiv</a>. In ICDE'23.

<small>+ denotes equal contribution, * denotes corresponding author</small>

## Introduction
Social recommender systems have drawn a lot of attention in many online web services, because of the incorporation of social information between users in improving recommendation results. Despite the significant progress made by existing solutions, we argue that current methods fall short in two limitations: (1) Existing social-aware recommendation models only consider collaborative similarity between items, how to incorporate item-wise semantic relatedness is less explored in current recommendation paradigms. (2) Current social recommender systems neglect the entanglement of the latent factors over heterogeneous relations (e.g., social connections, user-item interactions). Learning the disentangled representations with relation heterogeneity poses great challenge for social recommendation. In this work, we design a Disentangled Graph Neural Network (DGNN) with the integration of latent memory units, which empowers DGNN to maintain factorized representations for heterogeneous types of user and item connections. Additionally, we devise new memory-augmented message propagation and aggregation schemes under the graph neural architecture, allowing us to recursively distill semantic relatedness into the representations of users and items in a fully automatic manner. Extensive experiments on three benchmark datasets verify the effectiveness of our model by achieving great improvement over state-of-the-art recommendation techniques.

<img src='figs/framework.jpg' />

## Requirements
* python 3.7
* Pytorch 1.5+
* DGL 0.5.x
* numpy 1.17+

## Usage
Please unzip the datasets first, and use the following commands. If OOM occurs, use --n_layers 1 instead.
```
python main.py --dataset Ciao --data_path datasets/ciao/dataset.pkl --val_neg_path datasets/ciao/val_neg_samples.pkl --test_neg_path datasets/ciao/test_neg_samples.pkl
```
```
python main.py --dataset Epinions --data_path datasets/epinions/dataset.pkl --val_neg_path datasets/epinions/val_neg_samples.pkl --test_neg_path datasets/epinions/test_neg_samples.pkl
```
```
python main.py --dataset Yelp --data_path datasets/yelp/dataset.pkl --val_neg_path datasets/yelp/val_neg_samples.pkl --test_neg_path datasets/yelp/test_neg_samples.pkl
```
