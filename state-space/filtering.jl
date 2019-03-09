using Gen
using PyPlot
using Printf: @sprintf
import Random
import Distributions
using Statistics: median, mean
using JLD

const patches = PyPlot.matplotlib[:patches]

include("piecewise_normal.jl")
include("geometry.jl")

########################################
# variants of the HMM generative model #
########################################

@gen function lightweight_hmm(step::Int, path::Path, distances_from_start::Vector{Float64},
                              times::Vector{Float64}, speed::Float64,
                              noise::Float64, dist_slack::Float64)
    @assert step >= 1

    # walk path
    locations = Vector{Point}(undef, step)
    dist = @trace(normal(speed * times[1], dist_slack), (:dist, 1))
    locations[1] = walk_path(path, distances_from_start, dist)
    for t=2:step
        dist = @trace(normal(dist + speed * (times[t] - times[t-1]), dist_slack), (:dist, t))
        locations[t] = walk_path(path, distances_from_start, dist)
    end

    # generate noisy observations
    for t=1:step
        point = locations[t]
        @trace(normal(point.x, noise), (:x, t))
        @trace(normal(point.y, noise), (:y, t))
    end

    return locations
end

struct KernelState
    dist::Float64
    loc::Point
end

struct KernelParams
    times::Vector{Float64}
    path::Path
    distances_from_start::Vector{Float64}
    speed::Float64
    dist_slack::Float64
    noise::Float64
end

function dist_mean(prev_state::KernelState, params::KernelParams, t::Int)
    if t > 1
        prev_state.dist + (params.times[t] - params.times[t-1]) * params.speed
    else
        params.speed * params.times[1]
    end
end

@gen function lightweight_hmm_kernel(t::Int, prev_state::Any, params::KernelParams)
    # NOTE: if t = 1 then prev_state contains all NaNs
    dist = @trace(normal(dist_mean(prev_state, params, t), params.dist_slack), :dist)
    loc = walk_path(params.path, params.distances_from_start, dist)
    @trace(normal(loc.x, params.noise), :x)
    @trace(normal(loc.y, params.noise), :y)
    return KernelState(dist, loc)
end

# args: (num_steps::Int, init_state::KernelState, params::KernelParams)
lightweight_hmm_with_unfold = Unfold(lightweight_hmm_kernel)

@gen (static) function static_hmm_kernel(t::Int, prev_state::KernelState, params::KernelParams)
    # NOTE: if t = 1 then prev_state contains all NaNs
    dist = @trace(normal(dist_mean(prev_state, params, t), params.dist_slack), :dist)
    loc = walk_path(params.path, params.distances_from_start, dist)
    @trace(normal(loc.x, params.noise), :x)
    @trace(normal(loc.y, params.noise), :y)
    new_state = KernelState(dist, loc)
    return new_state
end

# args: (num_steps::Int, init_state::KernelState, params::KernelParams)
static_hmm = Unfold(static_hmm_kernel)


########################################
# analytically derived custom proposal #
########################################

function compute_custom_proposal_params(dt::Float64, prev_dist::Float64, noise::Float64, obs::Point,
                                        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
                                        path::Path, distances_from_start::Vector{Float64},
                                        speed::Float64, dist_slack::Float64)
    N = length(path.points)

    # Initialize parameters for truncated normal
    unnormalized_log_segment_probs = Vector{Float64}(undef, N+1)
    mus  = Vector{Float64}(undef, N+1)
    stds = Vector{Float64}(undef, N+1)

    # First segment
    mus[1] = prev_dist + dt * speed
    stds[1] = dist_slack
    unnormalized_log_segment_probs[1] = Distributions.logcdf(Distributions.Normal(mus[1], stds[1]), 0) + Distributions.logpdf(Distributions.Normal(path.start.x, noise), obs.x) + Distributions.logpdf(Distributions.Normal(path.start.y, noise), obs.y)
   
    # Last segment
    mus[N+1] = prev_dist + dt * speed
    stds[N+1] = dist_slack
    unnormalized_log_segment_probs[N+1] = Distributions.logccdf(Distributions.Normal(mus[N+1], stds[N+1]), distances_from_start[end]) +  Distributions.logpdf(Distributions.Normal(path.goal.x, noise), obs.x) + Distributions.logpdf(Distributions.Normal(path.goal.y, noise), obs.y)

    # Middle segments
    for i=2:N
        dx = path.points[i].x - path.points[i-1].x 
        dy = path.points[i].y - path.points[i-1].y
        dd = distances_from_start[i] - distances_from_start[i-1]

        prior_mu_d = mus[1] - distances_from_start[i-1]
        x_obs = obs.x - path.points[i-1].x 
        y_obs = obs.y - path.points[i-1].y

        posterior_mu_d = posterior_var_d * ((dx * x_obs + dy * y_obs) / (dd * noise^2) + prior_mu_d / dist_slack^2)

        mu_xy = prior_mu_d/dd .* [dx, dy]

        mus[i] = posterior_mu_d + distances_from_start[i-1]
        stds[i] = sqrt(posterior_var_d)
        unnormalized_log_segment_probs[i] = Distributions.logpdf(Distributions.MvNormal(mu_xy, posterior_covars[i]), [x_obs, y_obs]) + log(Distributions.cdf(Distributions.Normal(posterior_mu_d, stds[i]), dd) - Distributions.cdf(Distributions.Normal(posterior_mu_d, stds[i]), 0))
    end

    # You can think of the piecewise truncated normal as doing (1) categorical
    # draw, and (2) within that, a truncated normal draw.
    log_total_weight = logsumexp(unnormalized_log_segment_probs)
    log_normalized_weights = unnormalized_log_segment_probs .- log_total_weight
    probabilities = exp.(log_normalized_weights)

    return (probabilities, mus, stds, distances_from_start)
end


#####################################
# show path and observations on top #
#####################################

function render_hmm_trace(start::Point, stop::Point,
                path::Path,
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace,
                get_x::Function, get_y::Function, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    # set current axis
    sca(ax)

    # plot start and stop
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end

    # plot path lines
    for i=1:length(path.points)-1
        prev = path.points[i]
        next = path.points[i+1]
        plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
    end

    # plot measured locations
    if show_measurements
        measured_xs = [get_x(trace, i) for i=1:length(times)]
        measured_ys = [get_y(trace, i) for i=1:length(times)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end



######################
# show prior samples #
######################

const times = collect(range(0, stop=1, length=20))
const start_x = 0.1
const start_y = 0.1
const stop_x = 0.5
const stop_y = 0.5
const speed = 0.5
const noise = 0.02
const dist_slack = 0.2
const start = Point(start_x, start_y)
const stop = Point(stop_x, stop_y)

const path = Path(Point(0.1, 0.1), Point(0.5, 0.5), Point[Point(0.1, 0.1), Point(0.0773627, 0.146073), Point(0.167036, 0.655448), Point(0.168662, 0.649074), Point(0.156116, 0.752046), Point(0.104823, 0.838075), Point(0.196407, 0.873581), Point(0.390309, 0.988468), Point(0.408272, 0.91336), Point(0.5, 0.5)])
const distances_from_start = compute_distances_from_start(path)

function show_prior_samples()
    Random.seed!(0)

    # show samples from the lightweight model
    args = (length(times), path, distances_from_start, times, speed, noise, dist_slack)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(lightweight_hmm, args)
        render_hmm_trace(start, stop, path, times, speed, noise, dist_slack, trace, (tr, i) -> tr[(:x, i)], (tr, i) -> tr[(:y, i)], ax)
    end
    savefig("lightweight_hmm_prior_samples.png")

    # show samples from the static model
    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    args = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(static_hmm, (length(times), args...))
        render_hmm_trace(start, stop, path, times, speed, noise, dist_slack, trace, (tr, i) -> tr[i => :x], (tr, i) -> tr[i => :y], ax)
    end
    savefig("static_hmm_prior_samples.png")

end


###########################################
# evaluation harness for particle filters #
###########################################

struct Params
    times::Vector{Float64}
    speed::Float64
    dist_slack::Float64
    noise::Float64
    path::Path
end

struct PrecomputedPathData
    posterior_var_d::Float64
    posterior_covars::Vector{Matrix{Float64}}
    distances_from_start::Vector{Float64}
end

function PrecomputedPathData(params::Params)
    times = params.times
    speed = params.speed
    dist_slack = params.dist_slack
    noise = params.noise
    path = params.path

    distances_from_start = compute_distances_from_start(path)

    # posterior_var_d is the variance of the posterior on d' given x and y.
    # posterior_covars is a vector of 2x2 covariance matrices, representing the
    # covariance of x and y when d' has been marginalized out.
    posterior_var_d = dist_slack^2 * noise^2 / (dist_slack^2 + noise^2)
    posterior_covars = Vector{Matrix{Float64}}(undef, length(distances_from_start))
    for i = 2:length(distances_from_start)
        dd = distances_from_start[i] - distances_from_start[i-1]
        dx = path.points[i].x - path.points[i-1].x 
        dy = path.points[i].y - path.points[i-1].y
        posterior_covars[i] = [noise 0; 0 noise] .+ (dist_slack^2/dd^2 .* [dx^2 dx*dy; dy*dx dy^2])
    end

    PrecomputedPathData(posterior_var_d, posterior_covars, distances_from_start)
end

function evaluate_particle_filter(pf::Function, params::Params,
            measured_xs::Vector{Float64}, measured_ys::Vector{Float64},
            num_particles_list::Vector{Int}, num_reps::Int)

    precomputed = PrecomputedPathData(params)
    times = params.times
    speed = params.speed
    dist_slack = params.dist_slack
    noise = params.noise
    path = params.path

    results = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
        
            # run the particle filter
            lml = pf(measured_xs, measured_ys, num_particles, precomputed, path, times, speed, noise, dist_slack)

            # record results
            elapsed[rep] = Int(time_ns() - start) / 1e9
            lmls[rep] = lml
        end
        results[num_particles] = (lmls, elapsed)
    end
    return results
end


####################################
# static unfold + default proposal #
####################################

function static_default_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[1 => :x] = measured_xs[1]
    init_obs[1 => :y] = measured_ys[1]
    kernel_params = KernelParams(times, path,
                                 precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    args = (1, KernelState(NaN, Point(NaN, NaN)), kernel_params)
    state = initialize_particle_filter(static_hmm, args, init_obs, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, KernelState(NaN, Point(NaN, NaN)), kernel_params)
        argdiffs = (UnknownChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[step => :x] = measured_xs[step]
        obs[step => :y] = measured_ys[step]
        particle_filter_step!(state, args, argdiffs, obs)
    end
    lml = log_ml_estimate(state)
    return lml
end


###################################
# static unfold + custom proposal #
###################################

@gen (static) function static_fancy_proposal_init_inner(dt::Float64, noise::Float64, obs::Point,
                                        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
                                        path::Path, distances_from_start::Vector{Float64},
                                        speed::Float64, dist_slack::Float64)
    dist_params = compute_custom_proposal_params(
        dt, 0., noise, obs, posterior_var_d, posterior_covars,
        path, distances_from_start, speed, dist_slack)

    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), :dist)
end

static_fancy_proposal_init = call_at(static_fancy_proposal_init_inner, Int)

@gen (static) function static_fancy_proposal_step_inner(trace, step::Int, dt::Float64, noise::Float64, obs::Point,
                                        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
                                        path::Path, distances_from_start::Vector{Float64},
                                        speed::Float64, dist_slack::Float64)
    prev_dist = get_choices(trace)[step-1 => :dist]
    dist_params = compute_custom_proposal_params(
        dt, prev_dist, noise, obs, posterior_var_d, posterior_covars,
        path, distances_from_start, speed, dist_slack)

    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), :dist)
end

static_fancy_proposal_step = call_at(static_fancy_proposal_step_inner, Int)


function static_unfold_custom_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[1 => :x] = measured_xs[1]
    init_obs[1 => :y] = measured_ys[1]
    proposal_args = (times[1], noise,
                     Point(measured_xs[1], measured_ys[1]),
                     precomputed.posterior_var_d, precomputed.posterior_covars, path,
                     precomputed.distances_from_start, speed, dist_slack, 1)
    kernel_params = KernelParams(times, path, precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    args = (1, KernelState(NaN, Point(NaN, NaN)), kernel_params)
    state = initialize_particle_filter(static_hmm, args, init_obs,
                static_fancy_proposal_init, proposal_args, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, KernelState(NaN, Point(NaN, NaN)), kernel_params)
        argdiffs = (UnknownChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[step => :x] = measured_xs[step]
        obs[step => :y] = measured_ys[step]
        dt = step == 1 ? times[step] : times[step] - times[step-1]
        proposal_args = (step, dt, noise,
                         Point(measured_xs[step], measured_ys[step]),
                         precomputed.posterior_var_d, precomputed.posterior_covars,  path,
                         precomputed.distances_from_start, speed, dist_slack, step)
        particle_filter_step!(state, args, argdiffs, obs, static_fancy_proposal_step, proposal_args)
    end
    lml = log_ml_estimate(state)
    return lml
end


##################################
# lightweight + default proposal #
##################################

function lightweight_default_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[(:x, 1)] = measured_xs[1]
    init_obs[(:y, 1)] = measured_ys[1]
    args = (1, path, precomputed.distances_from_start, times, speed, noise, dist_slack)
    state = initialize_particle_filter(lightweight_hmm, args, init_obs, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, path, precomputed.distances_from_start, times, speed, noise, dist_slack)
        argdiffs = (UnknownChange(), NoChange(), NoChange(), 
            NoChange(), NoChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[(:x, step)] = measured_xs[step]
        obs[(:y, step)] = measured_ys[step]
        particle_filter_step!(state, args, argdiffs, obs)
    end
    lml = log_ml_estimate(state)
    return lml
end


#################################
# lightweight + custom proposal #
#################################

@gen function lightweight_fancy_proposal_init(dt::Float64, noise::Float64, obs::Point,
                                        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
                                        path::Path, distances_from_start::Vector{Float64},
                                        speed::Float64, dist_slack::Float64)
    dist_params = compute_custom_proposal_params(
        dt, 0., noise, obs, posterior_var_d, posterior_covars,
        path, distances_from_start, speed, dist_slack)
    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), (:dist, 1))
end


@gen function lightweight_fancy_proposal_step(trace, step::Int, dt::Float64, noise::Float64, obs::Point,
                                        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
                                        path::Path, distances_from_start::Vector{Float64},
                                        speed::Float64, dist_slack::Float64)
    @assert step > 1
    prev_dist = get_choices(trace)[(:dist, step-1)]
    dist_params = compute_custom_proposal_params(
        dt, prev_dist, noise, obs, posterior_var_d, posterior_covars,
        path, distances_from_start, speed, dist_slack)
    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), (:dist, step))
end

function lightweight_custom_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[(:x, 1)] = measured_xs[1]
    init_obs[(:y, 1)] = measured_ys[1]
    proposal_args = (times[1], noise,
                     Point(measured_xs[1], measured_ys[1]),
                     precomputed.posterior_var_d, precomputed.posterior_covars, path,
                     precomputed.distances_from_start, speed, dist_slack)
    kernel_params = KernelParams(times, path,
                                 precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    args = (1, path, precomputed.distances_from_start, times, speed, noise, dist_slack)
    state = initialize_particle_filter(lightweight_hmm, args, init_obs,
                lightweight_fancy_proposal_init, proposal_args, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, path, precomputed.distances_from_start, times, speed, noise, dist_slack)
        argdiffs = (UnknownChange(), NoChange(), NoChange(), 
            NoChange(), NoChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[(:x, step)] = measured_xs[step]
        obs[(:y, step)] = measured_ys[step]
        dt = times[step] - times[step-1]
        proposal_args = (step, dt, noise,
                         Point(measured_xs[step], measured_ys[step]),
                         precomputed.posterior_var_d, precomputed.posterior_covars,  path,
                         precomputed.distances_from_start, speed, dist_slack)
        particle_filter_step!(state, args, argdiffs, obs, lightweight_fancy_proposal_step, proposal_args)
    end
    lml = log_ml_estimate(state)
    return lml
end


#########################################
# lightweight unfold + default proposal #
#########################################

function lightweight_unfold_default_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[1 => :x] = measured_xs[1]
    init_obs[1 => :y] = measured_ys[1]
    kernel_params = KernelParams(times, path, precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    args = (1, KernelState(NaN, Point(NaN, NaN)), kernel_params)
    state = initialize_particle_filter(lightweight_hmm_with_unfold, args, init_obs, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, KernelState(NaN, Point(NaN, NaN)), kernel_params)
        argdiffs = (UnknownChange(), NoChange(), NoChange(), 
            NoChange(), NoChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[step => :x] = measured_xs[step]
        obs[step => :y] = measured_ys[step]
       particle_filter_step!(state, args, argdiffs, obs)
    end
    lml = log_ml_estimate(state)
    return lml
end

########################################
# lightweight unfold + custom proposal #
########################################

@gen function lightweight_unfold_fancy_proposal_init(dt::Float64, noise::Float64, obs::Point,
        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
        path::Path, distances_from_start::Vector{Float64},
        speed::Float64, dist_slack::Float64)
    dist_params = compute_custom_proposal_params(dt, 0., noise, obs,
                                                 posterior_var_d, posterior_covars, path, distances_from_start,
                                                 speed, dist_slack)
    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), 1 => :dist)
end

@gen function lightweight_unfold_fancy_proposal_step(trace, step::Int, dt::Float64, noise::Float64, obs::Point,
        posterior_var_d::Float64, posterior_covars::Vector{Matrix{Float64}},
        path::Path, distances_from_start::Vector{Float64},
        speed::Float64, dist_slack::Float64)
    @assert step > 1
    prev_dist = get_choices(trace)[step-1 => :dist]
    dist_params = compute_custom_proposal_params(dt, prev_dist, noise, obs,
                                                 posterior_var_d, posterior_covars, path, distances_from_start,
                                                 speed, dist_slack)
    @trace(piecewise_normal(dist_params[1], dist_params[2], dist_params[3], dist_params[4]), step => :dist)
end

function lightweight_unfold_custom_proposal_pf(measured_xs, measured_ys,
            num_particles, precomputed, path, times, speed, noise, dist_slack)
    ess_threshold = num_particles / 2
    init_obs = choicemap()
    init_obs[1 => :x] = measured_xs[1]
    init_obs[1 => :y] = measured_ys[1]
    proposal_args = (times[1], noise,
                     Point(measured_xs[1], measured_ys[1]),
                     precomputed.posterior_var_d, precomputed.posterior_covars, path,
                     precomputed.distances_from_start, speed, dist_slack)
    kernel_params = KernelParams(times, path,
                                 precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    kernel_params = KernelParams(times, path, precomputed.distances_from_start,
                                 speed, dist_slack, noise)
    args = (1, KernelState(NaN, Point(NaN, NaN)), kernel_params)
    state = initialize_particle_filter(lightweight_hmm_with_unfold, args, init_obs,
                lightweight_unfold_fancy_proposal_init, proposal_args, num_particles)
    for step=2:length(measured_xs)
        maybe_resample!(state, ess_threshold=ess_threshold, verbose=true)
        args = (step, KernelState(NaN, Point(NaN, NaN)), kernel_params)
        argdiffs = (UnknownChange(), NoChange(), NoChange(), 
            NoChange(), NoChange(), NoChange(), NoChange())
        obs = choicemap()
        obs[step => :x] = measured_xs[step]
        obs[step => :y] = measured_ys[step]
        dt = times[step] - times[step-1]
        proposal_args = (step, dt, noise,
                         Point(measured_xs[step], measured_ys[step]),
                         precomputed.posterior_var_d, precomputed.posterior_covars,  path,
                         precomputed.distances_from_start, speed, dist_slack)
        particle_filter_step!(state, args, argdiffs, obs, lightweight_unfold_fancy_proposal_step, proposal_args)
    end
    lml = log_ml_estimate(state)
    return lml
end

function plot_results(results::Dict, num_particles_list::Vector{Int}, label::String, color::String)
    median_elapsed = [median(results[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results[num_particles][1]) for num_particles in num_particles_list]
    plot(median_elapsed, mean_lmls, label=label, color=color)
end

function experiment()

    Random.seed!(0)

    # generate a path
    path = Path(Point(0.1, 0.1), Point(0.5, 0.5), Point[Point(0.1, 0.1), Point(0.0773627, 0.146073), Point(0.167036, 0.655448), Point(0.168662, 0.649074), Point(0.156116, 0.752046), Point(0.104823, 0.838075), Point(0.196407, 0.873581), Point(0.390309, 0.988468), Point(0.408272, 0.91336), Point(0.5, 0.5)])

    println("path:")
    println(path)

    # precomputation
    params = Params(times, speed, dist_slack, noise, path)
    precomputed = PrecomputedPathData(params)

    # generate ground truth locations and observations
    args = (length(times), path, precomputed.distances_from_start, times, speed, noise, dist_slack)
    trace = simulate(lightweight_hmm, args)
    assignment = get_choices(trace)
    measured_xs = [assignment[(:x, i)] for i=1:length(times)]
    measured_ys = [assignment[(:y, i)] for i=1:length(times)]
    actual_dists = [assignment[(:dist, i)] for i=1:length(times)]

    println("measured_xs:")
    println(measured_xs)

    println("measured_ys:")
    println(measured_ys)

    # parameters for particle filtering
    num_particles_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300]
    num_reps = 50

    # experiments with static model
    results_static_default_proposal = evaluate_particle_filter(static_default_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)
    results_static_custom_proposal = evaluate_particle_filter(static_unfold_custom_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)

    # experiments with lightweight model (no unfold)
    results_lightweight_default_proposal = evaluate_particle_filter(lightweight_default_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)
    results_lightweight_custom_proposal = evaluate_particle_filter(lightweight_custom_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)

    # experiments with unfold
    results_lightweight_unfold_default_proposal = evaluate_particle_filter(lightweight_unfold_default_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)
    results_lightweight_unfold_custom_proposal = evaluate_particle_filter(lightweight_unfold_custom_proposal_pf,
        params, measured_xs, measured_ys, num_particles_list, num_reps)

    save("results.jld",
        "results_static_default_proposal", results_static_default_proposal,
        "results_static_custom_proposal", results_static_custom_proposal,
        "results_lightweight_default_proposal", results_lightweight_default_proposal,
        "results_lightweight_custom_proposal", results_lightweight_custom_proposal,
        "results_lightweight_unfold_default_proposal", results_lightweight_unfold_default_proposal,
        "results_lightweight_unfold_custom_proposal", results_lightweight_unfold_custom_proposal)
end

# NOTE: you have to separate static gen function definitions from calling API
# methods on them with a top-level call to this function
Gen.load_generated_functions()

show_prior_samples()
experiment()
