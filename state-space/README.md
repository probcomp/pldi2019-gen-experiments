# Particle filtering in a nonlinear state space model

Here we evaluate several implementations of particle filtering in Gen as well as the particle filtering algorithm supported by Turing.
The state-space model is a hidden markov model with $T$ time steps, where the hidden states $d_t \in \mathbb{R}$ for $t=1\ldots T$ are the distances that an autonomous agent has moved along a piecewise linear path $p$ in the x-y plane.
The observations are pairs $(x_t, y_t)$ for $t=1\ldots T$.
