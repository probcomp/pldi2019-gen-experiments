# Experiments for Gen, PLDI 2019

This repository contains the code for figures and experiments appearing in

> Marco F. Cusumano-Towner, Feras A. Saad, Alex Lew, and Vikash K. Mansinghka.
> 2019. Gen: A General-Purpose Probabilistic Programming System with
> Programmable Inference. To Appear In Proceedings of 39th ACM SIGPLAN
> Conference on Programming Language Design and Implementation (PLDI'19). ACM,
> New York, NY, USA

## Structure of the repository

- [example](./example) contains the code for the tutorial in Figure 2.
- [regression](./regression) contains the code for the robust Bayesian regression benchmark in Section 7.1.
- [gp](./gp) contains the code for the Gaussian process structure benchmark in Section 7.2.
- [algorithmic-model](./algorithmic-model) contains the code for the algorithmic model of an autonomous agent in Section 7.3.
- [state-space](./state-space) contains the code for the nonlinear state-space model in Section 7.4.
- [pose](./pose) contains the code of the pose estimation application in Section 7.5.

## Basic instructions to set up the Julia environment

1. Download and install Julia v1.1

2. Clone `git@github.com:probcomp/pldi2019-gen-experiments`

3. Run `export JULIA_PROJECT=/path/to/pldi2019-gen-experiments`, where
   `/path/to` should be the prefix of the absolute path of this repository on
   your local disk.

4. Set the environment variable `JULIA_PROJECT` to the full path of this repository.

5. Install dependencies using `julia -e 'using Pkg; Pkg.instantiate()'
