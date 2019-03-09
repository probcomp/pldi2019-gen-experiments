using Turing

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

prior = [0.2, 0.3, 0.5]

emission_dists = [
    0.1 0.2 0.7;
    0.2 0.7 0.1;
    0.7 0.2 0.1
]'

transition_dists = [
    0.4 0.4 0.2;
    0.2 0.3 0.5;
    0.9 0.05 0.05
]'

num_steps = 4

observations = [1, 1, 2, 3]

ml = hmm_forward_alg(prior, emission_dists, transition_dists, observations)
println("true log ml: $(log(ml))")

function logsumexp(arr::Vector{Float64})
    max_arr = maximum(arr)
    max_arr + log(sum(exp.(arr .- max_arr)))
end

import Distributions

function run_handcoded_pf(num_particles::Int)

    # allocate
    particles = Vector{Int}(undef, num_particles)
    new_particles = Vector{Int}(undef, num_particles)
    parents = Vector{Int}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)

    # initialize log marginal likelihood estimate
    lml = 0.0

    # initialize particles by sampling from the prior
    Distributions.rand!(Distributions.Categorical(prior), particles)
    for i=1:num_particles
        log_weights[i] = log(emission_dists[particles[i], observations[1]])
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
            log_weights[i] = log(emission_dists[new_particles[i], observations[t]])
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

# run particle filter
function run_turing_pf(num_particles::Int)
    spl = Turing.Sampler(Turing.SMC(num_particles, Turing.resample_multinomial, 0.5, Set(), 0))
    particles = Turing.ParticleContainer{Turing.Trace}(
        hmm(prior, emission_dists, transition_dists, observations))
    push!(particles, spl.alg.n_particles, spl, Turing.VarInfo())
    while Libtask.consume(particles) != Val{:done}
        ess = Turing.effectiveSampleSize(particles)
        #if ess <= spl.alg.resampler_threshold * length(particles)
        Turing.resample!(particles,spl.alg.resampler)
        #end
    end
    particles.logE
end


println("Handcoded:")
for num_particles in [1, 100, 1000, 10000, 100000, 1000000]
    for i=1:10
        lml = run_handcoded_pf(num_particles)
        println("lml estimate with $num_particles particles: $lml")
    end
end

println("Turing:")
for num_particles in [1, 100, 1000, 10000, 100000]
    for i=1:10
        lml = run_turing_pf(num_particles)
        println("lml estimate with $num_particles particles: $lml")
    end
end




