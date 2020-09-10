# Neural Network Deconvolution (NND)

## Introduction

NND is a toolkit that unmixes bulk tumor samples. Given a non-negative bulk RNA expression matrix `B \in R_+^{m x n}`, where each row `i` is a gene, each column `j` is a tumor sample, our goal is to infer an expression profile matrix `C \in R_+^{m x k}`, where each column `l` is a cell community, and a fraction matrix `F \in R_+^{k x n}`, such that:
```
  B ~= C F. 
```
To be more specific, NND solves the following problem:
```
min_{C, F} || B - C F ||_{Fr}^2, 

      s.t. C_{il} >= 0,              i=1,...,m, l=1,...,k,

           F_{lj} >= 0,              l=1,...,k, j=1,...,n,

           \sum_{l=1}^{k} F_{lj} = 1,           j=1,...,n.
```

NND transfers the above problem equivalently into optimizating a neural network, which can be solved using through gradient descent.

NND has the following functions:
* `compress_module`: Integrate gene module knowledge to reduce noise.
* `estimate_number`: Estimate the number of cell populations automatically.
* `estimate_clones`: Utilize core NND algorithm to unmix the cell populations.
* `estimate_marker`: Estimate other biomarkers of cell populations given bulk marker data.

## Prerequisites

The code runs on Python 3, and requires `cvxopt` and PyTorch. Most other packages are available in the Anaconda.

## Usage

You can find a brief tutorial with code and output in `tutorial.ipynb`.


## Citation

If you find NND helpful, please cite the following paper: 
Yifeng Tao, Haoyun Lei, Adrian V. Lee, Jian Ma, and Russell Schwartz. [**Neural network deconvolution method for resolving pathway-level progression of tumor clonal expression programs with application to breast cancer brain metastases**](https://www.frontiersin.org/articles/10.3389/fphys.2020.01055/full). *Frontiers in Physiology*, 11:1055. 2020.
```
@article{tao2020nnd,
  title = {Neural Network Deconvolution Method for Resolving Pathway-Level Progression of Tumor Clonal Expression Programs with Application to Breast Cancer Brain Metastases},
  author = {Tao, Yifeng and Lei, Haoyun and Lee, Adrian V. and Ma, Jian and Schwartz, Russell},
  journal = {Frontiers in Physiology},
  volume = {11},
  pages = {1055},
  year = {2020},
  url = {https://www.frontiersin.org/article/10.3389/fphys.2020.01055},
  doi = {10.3389/fphys.2020.01055},
  issn = {1664-042X},
}
```

We further developed the NND method into [Robust and Accurate Deconvolution (RAD)](https://github.com/CMUSchwartzLab/RAD), which follows the same APIs to NND, but is faster and more accurate compared with NND and competing algorithms.