using Turing
using Statistics: mean

function logsumexp(arr::AbstractArray{T}) where {T <: Real}
    max_arr = maximum(arr)
    max_arr + log(sum(exp.(arr .- max_arr)))
end


#####################
# forward algorithm #
#####################

function hmm_forward_alg(prior::Vector{Float64},
        emission_dists::AbstractArray{Float64,2},
        transition_dists::AbstractArray{Float64,2},
        emissions::Vector{Int})

    marg_lik = 1.
    alpha = prior # p(z_1)
    for i=2:length(emissions)

        # p(z_{i-1} , y_{i-1} | y_{1:i-2}) for each z_{i-1}
        prev_posterior = alpha .* emission_dists[emissions[i-1], :] 

        # p(y_{i-1} | y_{1:i-2})
        denom = sum(prev_posterior) 

        # p(z_{i-1} | y_{1:i-1})
        prev_posterior = prev_posterior / denom 

        # p(z_i | y_{1:i-1})
        alpha = transition_dists * prev_posterior

        # p(y_{1:i-1})
        marg_lik *= denom
    end
    prev_posterior = alpha .* emission_dists[emissions[end], :] 
    denom = sum(prev_posterior) 
    marg_lik *= denom
    marg_lik
end


##################################
# forward algorithm in log space #
##################################

function hmm_forward_alg_logspace(prior::Vector{Float64},
        emission_dists::AbstractArray{Float64,2},
        transition_dists::AbstractArray{Float64,2},
        emissions::Vector{Int})

    log_marg_lik = 0.
    log_alpha = log.(prior) # p(z_1)
    for i=2:length(emissions)

        # p(z_{i-1} , y_{i-1} | y_{1:i-2}) for each z_{i-1}
        log_prev_posterior = log_alpha .+ log.(emission_dists[emissions[i-1], :])

        # p(y_{i-1} | y_{1:i-2})
        log_denom = logsumexp(log_prev_posterior) 

        # p(z_{i-1} | y_{1:i-1})
        log_prev_posterior = log_prev_posterior .- log_denom 

        # p(z_i | y_{1:i-1})
        @assert length(log_alpha) == length(prior)
        for state=1:length(prior)
            log_alpha[state] = logsumexp(log.(transition_dists[state,:]) .+ log_prev_posterior)
        end

        # p(y_{1:i-1})
        log_marg_lik += log_denom
    end
    log_prev_posterior = log_alpha .+ log.(emission_dists[emissions[end], :] )
    log_denom = logsumexp(log_prev_posterior) 
    log_marg_lik += log_denom
    log_marg_lik
end


#############################
# handcoded particle filter #
#############################

import Distributions
function run_handcoded_pf(num_particles::Int, prior, emission_dists, transition_dists, observations)

    # allocate
    particles = Vector{Int}(undef, num_particles)
    new_particles = Vector{Int}(undef, num_particles)
    parents = Vector{Int}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)

    # initialize log marginal likelihood estimate
    lml = 0.0

    # initialize particles by sampling from the prior
    Distributions.rand!(Distributions.Categorical(prior), particles)

    # initialize log_weights with the likelihood of the first observation
    for i=1:num_particles
        log_weights[i] = log(emission_dists[observations[1], particles[i]])
    end

    for t=2:length(observations)

        # increment log marginal likelihood estimate
        lml += (logsumexp(log_weights) - log(num_particles))

        # resample (sample the parents)
        weights = exp.(log_weights .- logsumexp(log_weights))
        Distributions.rand!(Distributions.Categorical(weights), parents)

        # extend the particles
        for i=1:num_particles
            new_particles[i] = rand(Distributions.Categorical(
                transition_dists[:,particles[parents[i]]]))
            log_weights[i] = log(emission_dists[observations[t], new_particles[i]])
        end

        # update the particles
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end

    # increment log marginal likelihood estimate
    lml += (logsumexp(log_weights) - log(num_particles))

    lml
end


####################################
# model and particle filter Turing #
####################################

@model hmm(prior, emission_dists, transition_dists, observations) = begin
    num_steps = length(observations)
    hiddens = Vector{Int}(undef, num_steps)
    hiddens[1] ~ Categorical(prior)
    observations[1] ~ Categorical(emission_dists[:,hiddens[1]])
    for t=2:num_steps
        hiddens[t] ~ Categorical(transition_dists[:,hiddens[t-1]])
        observations[t] ~ Categorical(emission_dists[:,hiddens[t]])
    end
end

function run_turing_pf(num_particles::Int, prior, emission_dists, transition_dists, observations)
    resampler_threshold = 1.0 # always resample
    alg = Turing.SMC(num_particles, Turing.resample_multinomial, resampler_threshold, Set(), 0)
    model = hmm(prior, emission_dists, transition_dists, observations)

    # adapted from https://github.com/TuringLang/Turing.jl/blob/6d48b96b03617f4410fd90becd7fb3a6c0d7e7e8/src/inference/smc.jl#L75-L86
    spl = Turing.Sampler(alg)

    particles = Turing.ParticleContainer{Turing.Trace}(model)
    push!(particles, spl.alg.n_particles, spl, Turing.VarInfo())

    while Libtask.consume(particles) != Val{:done}
        ess = Turing.effectiveSampleSize(particles)
        if ess <= spl.alg.resampler_threshold * length(particles)
            Turing.resample!(particles,spl.alg.resampler)
        end
    end

    # get log marginal likelihood (evidence) estimate
    # Q: is there a different intended way to get the log marginal likelihood estimate ?
    particles.logE
end


##################
# do experiments #
##################

prior = [0.2, 0.3, 0.5]

# emission_dists[obs, state] is the probability of observation `obs` arising from state `state`
emission_dists = [
    0.1 0.2 0.7;
    0.2 0.7 0.1;
    0.7 0.2 0.1
]'

# transition_dists[new_state, prev_state] is the probability of transitioning from 'prev_state' to 'new_state'
transition_dists = [
    0.4 0.4 0.2;
    0.2 0.3 0.5;
    0.9 0.05 0.05
]'

observations = [1, 1, 2, 3]

# compute ground truth log marginal likelihood
ground_truth_lml = log(hmm_forward_alg(prior, emission_dists, transition_dists, observations))
println("ground truth log ml: $(ground_truth_lml)")

ground_truth_lml_logspace = hmm_forward_alg_logspace(prior, emission_dists, transition_dists, observations)
println("ground truth log ml (computed in logspace): $(ground_truth_lml_logspace)")


import Random; Random.seed!(1)

num_reps = 100
num_particles_list = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

# run experiments with handcoded PF
handcoded_avg_lml = []
for num_particles in num_particles_list
    println(num_particles)
    lmls = [run_handcoded_pf(num_particles, prior, emission_dists, transition_dists, observations) for _=1:num_reps]
    push!(handcoded_avg_lml, mean(lmls))
end

# run experiments with Turing PF
turing_avg_lml = []
for num_particles in num_particles_list
    println(num_particles)
    lmls = [run_turing_pf(num_particles, prior, emission_dists, transition_dists, observations) for _=1:num_reps]
    push!(turing_avg_lml, mean(lmls))
end

# plot results
using PyPlot
figure(figsize=(6,6))
plot(num_particles_list, handcoded_avg_lml, marker="o", color="blue", label="handcoded PF")
plot(num_particles_list, turing_avg_lml, marker="o", color="red", label="Turing PF")
xlim = gca()[:get_xlim]()
ylim = gca()[:get_ylim]()
plot(xlim, [ground_truth_lml_logspace, ground_truth_lml_logspace], linestyle="--", color="black", label="ground truth")
gca()[:set_xscale]("log")
gca()[:set_ylim](ylim)
legend()
tight_layout()
savefig("results.png")
