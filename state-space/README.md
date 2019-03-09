# Particle filtering in a nonlinear state space model

Here we evaluate several implementations of particle filtering in Gen as well as the particle filtering algorithm supported by Turing.
The state-space model is a hidden markov model with $N$ time steps, where the hidden states $d_i \in \mathbb{R}$ for $i=1\ldots N$ are the distances that an autonomous agent has moved along a piecewise linear path $m$ in the x-y plane at given times $t_i \ge 0$.
The corresponding observations are pairs $(x_i, y_i)$ for $i=1\ldots N$.
There is a given nominal speed $v \ge 0$, and a given variability $\sigma_d$ in distance traveled and a given measurement noise $\sigma_m$.
Specifically the model is:

- $d_1 \sim \mathcal{N}(v t_1, \sigma_d)$ and $d_i \sim \mathcal{N}(d_{i-1} + v (t_i - t_{i-1}), \sigma_d)$ for $i=2\ldots N$.

- $x_i \sim \mathcal{N}(f_x(m, d_i), \sigma_m)$ and $y_i \sim \mathcal{N}(f_y(m, d_i), \sigma_m)$ for $i=1\ldots N$.

where $f_x(m, d)$ for path $m$ and distance $d$ is the x-coordinate of the point along the path $m$ found by walking a distance of $d$ along the path, and similarly for $f_y$.
Note that for $d$ that exceed the length of the path, $(f_x(m, d), f_y(m, d))$ is the terminal point on the path.
Note that although the dynamics model is linear-Gaussian, the measurement model is piecewise-linear Gaussian.

An example path $m$ is shown as in grey below, and an example data set of $(x_i, y_i)$ pairs are overlaid.
The first point in the path is highlighted in blue, and the terminal point is shown in red:

<img src="example.png" alt="example path and observations" width="250">

We will be evaluating the particle filters' ability to estimate the marginal likelihood of the observed $(x_i, y_i)$ for a given path $m$, using the the log marginal likelihood estimates returned by the particle filters as a function of the number of particles.
As we increase the number of particles, the log marginal likelihood estimates typically increase and approach the true log marginal likelihood.

We evaluate two particle filtering algorithms---one that uses the dynamics as the proposal for each $d_i$ (a *generic* proposal), and one which uses a *custom* proposal that takes the measurement $(x_i, y_i)$ into account when proposing $d_i$.
The custom proposal requires some manual derivation, and requires more code to implement, but the resulting particle filter is be significantly more efficient than the particle filter with the generic proposal.

## The generic proposal

The generic proposal for $i > 1$ is $d_i \sim \mathcal{N}(\cdot; d_{i-1} + v (t_i - t_{i-1}), \sigma_d)$

## The custom proposal

For the custom proposal, we will use the conditional distribution on $d_i$ given $d_{i-1}$ and $(x_i, y_i)$ under the generative model.
The density of the conditional distribution is given by:
$p(d_i | d_{i-1}, x_i, y_i) \propto g(d_i) := p(d_i | d_{i-1}) p(x_i | d_i) p(y_i | d_i)$.
The first factor is given by a normal distribution probability density function with mean $d_{i-1} + v (t_i - t_{i-1})$ and standard deviation $\sigma_d$.
$$p(d_i | d_{i-1}) = \frac{1}{\sqrt{2 \pi \sigma_d^2}} \exp{\left(-\frac{(d_i - d_{i-1} - v (t_i - t_{i-1}))^2}{2 \sigma_d^2}\right)}$$
The second factor is given by:
$$p(x_i | d_i) = \frac{1}{\sqrt{2 \pi \sigma_m^2}} \exp{\left(-\frac{(x_i - f_x(m, d_i))^2}{2 \sigma_m^2}\right)}$$
Similarly, the third factor is:
$$p(y_i | d_i) = \frac{1}{\sqrt{2 \pi \sigma_m^2}} \exp{\left(-\frac{(y_i - f_y(m, d_i))^2}{2 \sigma_m^2}\right)}$$
The normalizing constant is given by an integral:
$$Z = \int_{-\infty}^{\infty} g(d_i)  d d_i$$
We break the interval into pieces $(\infty, D_1), [D_1, D_2), \ldots [D_{K-1}, D_K), [D_K, \infty)$ where $K$ is the number of points that define the path $m$, including the first and and last points on the path (and where $D_1 := 0$):
$$Z = \int_{-\infty}^{D_1} g(d_i) d d_i + \sum_{k=1}^{K-1} \int_{D_k}^{D_{k+1}} g(d_i) d d_i + \int_{D_K}^{\infty} g(d_i) d d_i$$
Because, within each interval, the model is a linear Gaussian state space model (the functions $f_x$ and $f_y$ are linear in $d_i$ within an interval), we can analytically compute each of these $K+1$ terms, denoted $Z_k$ for $k=1\ldots K+1$.
The probability that $d_i$ lies in interval $k$ is given by $Z_k / Z$.
Conditioned on the choice of interval, $d_i$ has a normal distribution.
Therefore, our custom proposal distribution on $d_i$ takes the form of a *piecewise normal distribution*.
