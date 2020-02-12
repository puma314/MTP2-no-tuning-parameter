# Introduction

This is the open-source code for the algorithms and experiments in our paper [Learning High-dimensional Gaussian Graphical Models under Total Positivity without Adjustment of Tuning Parameters](https://arxiv.org/abs/1906.05159).

# Algorithm 1 of paper

Algorithm 1 of the paper is implemented in `main_algorithm.py` in this repository. Given samples `X` from a multivariate Gaussian distribution, use `no_tuning_parameters(X)` in `main_algorithm.py` to get an estimate of the 0-pattern of the underlying precision matrix omega according to Algorithm 1 of our paper.

## Other Algorithms for comparison

Other algorithms that we compare to in our paper are contained in `comparison_algorithms.py`. 


