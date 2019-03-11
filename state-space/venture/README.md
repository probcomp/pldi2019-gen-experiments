# Venture filtering experiment

First clone the [Venturecxx repository](https://github.com/probcomp/Venturecxx):
```
git clone git@github.com:probcomp/Venturecxx.git
```

Then install Venture and some other dependencies into a virtual environment:
```
virtualenv -p python2 env
pip install Venturecxx
pip install numpy matplotlib scipy pandas requests flask 
```

Then, run the experiment script:
```
python run_experiment.py
```

This will take a while to run, and will consume significant memory.
Note that `extensions.py` is currently set up to use 16 processes.
You should customize this to the number of cores on your machine.

## Obtaining log marginal likelihood estimates from Venture

Let $w_{t,i}$ denote the incremental weight of particle $i$ at time step $t=0,\ldots,T-1$ where $T$ is the number of time steps at which measurements are taken, and where $i=1,\ldots,N$.
These weights are given by the likelihoods of the observations at each time step:

$$w_{t,i} := p(x_t^{(i)}, y_t^{(i)} | d_t^{(i)})$$

Assuming resampling is done at every step, the marginal likelihood estimate is given by:

$$\hat{p}(\mathbf{x}, \mathbf{y}) = \prod_{t=0}^{T-1} \frac{1}{N} \sum_{i=1}^N w_{t,i}$$

In log-space this is:

$$\log \hat{p}(\mathbf{x}, \mathbf{y}) = \sum_{t=0}^{T-1} \log \frac{1}{N} \sum_{i=1}^N w_{t,i}$$

Venture keeps track of a 'log weight' internally for each of the particles.
We will denote these log weights by $W_{t,i}$.
These log weights are initialized to zero:

$$W_{-1,i}' = 0$$

With each block of observations at $t=0, \ldots, T-1$, Venture increments the 'log weight' of each particle by the log incremental weight:

$$W_{t,i} = W_{t-1,i}' + \log w_{t,i}$$

At each resampling step, Venture sets the 'log weight' of each particle to the same value:

$$W_{t,i}' := \log \frac{1}{N} \sum_{i=1}^N \exp{W_{t,i}}$$

We claim that we can compute the log marginal likelihood estimate from the final 'log weights' that are provided by Venture through the function `particle_log_weights()`:

$$\log \hat{p}(\mathbf{x}, \mathbf{y}) = \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}}$$

It suffices to show that:
$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} = \sum_{t=0}^{T-1} \log \frac{1}{N} \sum_{i=1}^N w_{t,i}$$

**Base case.**
We start with the base case of $T = 1$:

$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{0,i}} =  \log \frac{1}{N} \sum_{i=1}^N w_{0,i}$$

This follows from:

$$W_{0,i} = \log w_{0,i}$$

**Induction step.**
For some $T > 0$ we assume that the inductive hypothesis holds:

$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} = \sum_{t=0}^{T-1} \log \frac{1}{N} \sum_{i=1}^N w_{t,i}$$

and we seek to prove it for $T+1$:

$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T,i}} = \sum_{t=0}^{T} \log \frac{1}{N} \sum_{i=1}^N w_{t,i}$$

We break down the right-hand side into two terms:

$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T,i}} = \left(\sum_{t=0}^{T-1} \log \frac{1}{N} \sum_{i=1}^N w_{t,i}\right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$

$$\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T,i}} = \left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$

We now expand the $W_{T,i}$ on the left-hand side:

$$\log \frac{1}{N} \sum_{i=1}^N \exp{(W_{T-1,i}' + \log w_{T,i})} = \left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$

$$\log \frac{1}{N} \sum_{i=1}^N \exp{\left(\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} + \log w_{T,i}\right)} = \left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$

$$\log \left( \exp{\left(\log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}}\right)} \frac{1}{N} \sum_{i=1}^N w_{T,i}\right) = \left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$

$$\left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right) = \left( \log \frac{1}{N} \sum_{i=1}^N \exp{W_{T-1,i}} \right) + \left(\log \frac{1}{N} \sum_{i=1}^N w_{T,i}\right)$$
