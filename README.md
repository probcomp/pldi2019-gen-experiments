# pldi2019-gen-experiments
Experiments for PLDI 2019 submission on Gen

Basic instructions to run the experiments.

1. Download and install Julia v0.7

2. Clone `git@github.com:probcomp/pldi2019-gen-experiments`

3. Clone `git@github.com:probcomp/Gen`
    - run `cd Gen`.
    - run `git checkout 20181022-marcoct-pcfg`.

4. Clone `git@github.com:JuliaCollections/FunctionalCollections.jl`
    - run `cd FunctionalCollections.jl`.
    - run `checkout b261b9daa8ea438f1a63c78fe10d722ba581583c`.

5. Run `export JULIA_PROJECT=/path/to/pldi2019-gen-experiments`, where
    `/path/to` should be the prefix of the absolute path on your local disk.

6. Run `cd pldi2019-gen-experiments`
    - run `julia`
    - type `]` to enter the `pkg>` shell
    - run `develop /path/to/Gen`.
    - run `develop /path/to/FunctionalCollections.jl`

7. Configure `PyPlot` as follows
     - run `julia`
     - run `ENV["PYTHON"]=""`
     - type `]` to enter the `pkg>` shell
     - run `build PyCall`

8. Make sure that `JULIA_PROJECT` is set to the full path of
    `pldi2019-gen-experiments` when using this repo going forward.

9. Test your installation worked by running an experiment, e.g.

        $ cd pldi2019-gen-experiments/gp

        $ ./pipeline.jl --n-test=20 \
            --nprobe-held-in=2 \
            --npred-held-in=1 \
            --shortname=nVzAI \
            --iters=5 \
            --epochs=1000 \
            --chains=1 \
            ./resources/matlab_timeseries/01-airline.csv \
            incremental
