using Gen
using PyPlot
using Printf: @sprintf
import Random
import Distributions
using Statistics: median, mean

include("scenes.jl")
include("path_planner.jl")

# depends on: scenes.jl, path_planner.jl


#####################################
# lightweight implementation of hmm #
#####################################

@gen function lightweight_hmm(step::Int, path::Path, distances_from_start::Vector{Float64},
                              times::Vector{Float64}, speed::Float64,
                              noise::Float64, dist_slack::Float64)
    @assert step >= 1

    # walk path
    locations = Vector{Point}(undef, step)
    dist = @addr(normal(speed * times[1], dist_slack), (:dist, 1))
    locations[1] = walk_path(path, distances_from_start, dist)
    for t=2:step
        dist = @addr(normal(dist + speed * (times[t] - times[t-1]), dist_slack), (:dist, t))
        locations[t] = walk_path(path, distances_from_start, dist)
    end

    # generate noisy observations
    for t=1:step
        point = locations[t]
        @addr(normal(point.x, noise), (:x, t))
        @addr(normal(point.y, noise), (:y, t))
    end

    return locations
end

#################################
# piecewise normal distribution #
#################################

struct PieceWiseNormal <: Gen.Distribution{Float64} end
const piecewise_normal = PieceWiseNormal()

# piece 1 has interval (-Inf, boundaries[1])
# piece 2 has interval (boundaries[1], boundaries[2])
# ..
# piece n has interval (boundaries[n-1], boundaries[n])
# piece n+1 has interval (boundaries[n], Inf)

function Gen.random(::PieceWiseNormal, probabilities::Vector{Float64}, mus::Vector{Float64}, stds::Vector{Float64}, boundaries::Vector{Float64})
    n_segments = length(probabilities)
    @assert n_segments == length(mus)
    @assert n_segments == length(stds)
    @assert n_segments == length(boundaries)+1
    idx = categorical(probabilities)
    dist = Distributions.TruncatedNormal(mus[idx], stds[idx], idx == 1 ? -Inf : boundaries[idx-1], idx == n_segments ? Inf : boundaries[idx])
    rand(dist)
end

function Gen.logpdf(::PieceWiseNormal, x::Float64, probabilities::Vector{Float64}, mus::Vector{Float64}, stds::Vector{Float64}, boundaries::Vector{Float64})
    n_segments = length(probabilities)
    @assert n_segments == length(mus)
    @assert n_segments == length(stds)
    @assert n_segments == length(boundaries)+1
    log_probs = Vector{Float64}(undef, n_segments)
    for idx=1:length(probabilities)
        dist = Distributions.TruncatedNormal(mus[idx], stds[idx], idx == 1 ? -Inf : boundaries[idx-1], idx == n_segments ? Inf : boundaries[idx])
        log_probs[idx] = log(probabilities[idx]) + Distributions.logpdf(dist, x)
    end
    logsumexp(log_probs)
end



#######################################
# custom proposal for lightweight hmm #
#######################################

@gen function lightweight_fancy_step_proposal(step::Int, prev_dist::Float64, noise :: Float64, obs :: Point,
                                        posterior_var_d :: Float64, posterior_covars :: Vector{Matrix{Float64}},
                                        path :: Path, distances_from_start :: Vector{Float64},
                                        speed::Float64, times::Vector{Float64}, dist_slack::Float64)
    N = length(path.points)

    # Initialize parameters for truncated normal
    unnormalized_log_segment_probs = Vector{Float64}(undef, N+1)
    mus  = Vector{Float64}(undef, N+1)
    stds = Vector{Float64}(undef, N+1)

    dt = step == 1 ? times[step] : times[step] - times[step-1]

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

    # Now do a log categorical draw,
    # and within that, do a truncated normal sample.
    log_total_weight = logsumexp(unnormalized_log_segment_probs)
    log_normalized_weights = unnormalized_log_segment_probs .- log_total_weight
    probabilities = exp.(log_normalized_weights)
    @addr(piecewise_normal(probabilities, mus, stds, distances_from_start), (:dist, step))
end

##################################
# compiled implementation of hmm #
##################################

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

@compiled @gen function kernel(t::Int, prev_state::KernelState, params::KernelParams)
    # NOTE: if t = 1 then prev_state will have all NaNs
    dist::Float64 = @addr(normal(dist_mean(prev_state, params, t), params.dist_slack), :dist)
    loc::Point = walk_path(params.path, params.distances_from_start, dist)
    @addr(normal(loc.x, params.noise), :x)
    @addr(normal(loc.y, params.noise), :y)
    return KernelState(dist, loc)::KernelState
end

# construct a hidden markov model generator
# args: (num_steps::Int, init_state::KernelState, params::KernelParams)
compiled_hmm = markov(kernel)


#############################
# proposal for compiled hmm #
#############################

@gen function compiled_fancy_proposal_inner(dt::Float64, prev_dist::Float64, noise :: Float64, obs :: Point,
                                        posterior_var_d :: Float64, posterior_covars :: Vector{Matrix{Float64}},
                                        path :: Path, distances_from_start :: Vector{Float64},
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

    # Now do a log categorical draw,
    # and within that, do a truncated normal sample.
    log_total_weight = logsumexp(unnormalized_log_segment_probs)
    log_normalized_weights = unnormalized_log_segment_probs .- log_total_weight
    probabilities = exp.(log_normalized_weights)
    @addr(piecewise_normal(probabilities, mus, stds, distances_from_start), :dist)
end

compiled_fancy_proposal = at_dynamic(compiled_fancy_proposal_inner, Int)


#############################
# define some fixed context #
#############################

function make_scene()
    scene = Scene(0, 1, 0, 1) 
    add!(scene, Tree(Point(0.30, 0.20), size=0.1))
    add!(scene, Tree(Point(0.83, 0.80), size=0.1))
    add!(scene, Tree(Point(0.80, 0.40), size=0.1))
    horiz = 1
    vert = 2
    wall_height = 0.30
    wall_thickness = 0.02
    walls = [
        Wall(Point(0.20, 0.40), horiz, 0.40, wall_thickness, wall_height)
        Wall(Point(0.60, 0.40), vert, 0.40, wall_thickness, wall_height)
        Wall(Point(0.60 - 0.15, 0.80), horiz, 0.15 + wall_thickness, wall_thickness, wall_height)
        Wall(Point(0.20, 0.80), horiz, 0.15, wall_thickness, wall_height)
        Wall(Point(0.20, 0.40), vert, 0.40, wall_thickness, wall_height)]
    for wall in walls
        add!(scene, wall)
    end
    return scene
end

const scene = make_scene()
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


####################
# trace renderings #
####################

# the only difference between rendering the lightweight and compiled versions
# how the locations are extracted from the trace, and the addresses of the
# measurements.

function render_lightweight_hmm_trace(scene::Scene, start::Point, stop::Point,
                maybe_path::Nullable{Path},
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    # set current axis
    sca(ax)

    # render obstacles
    render(scene, ax)

    # plot start and stop
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end

    # plot path lines
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    locations = get_call_record(trace).retval
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    assignment = get_assignment(trace)
    if show_measurements
        measured_xs = [assignment[(:x, i)] for i=1:length(locations)]
        measured_ys = [assignment[(:y, i)] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end

function render_compiled_hmm_trace(scene::Scene, start::Point, stop::Point,
                maybe_path::Nullable{Path},
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    # set current axis
    sca(ax)

    # render obstacles
    render(scene, ax)

    # plot start and stop
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end

    # plot path lines
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    states = get_call_record(trace).retval
    locations = Point[state.loc for state in states]
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    assignment = get_assignment(trace)
    if show_measurements
        measured_xs = [assignment[i => :x] for i=1:length(locations)]
        measured_ys = [assignment[i => :y] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end


######################
# show prior samples #
######################

function show_prior_samples()
    Random.seed!(0)

    # generate a path
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)
    distances_from_start = compute_distances_from_start(path)

    # show samples from the lightweight model
    model_args = (length(times), path, distances_from_start, times, speed, noise, dist_slack)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(lightweight_hmm, model_args)
        render_lightweight_hmm_trace(scene, start, stop, maybe_path, times, speed, noise, dist_slack, trace, ax)
    end
    savefig("lightweight_hmm_prior_samples.png")

    # show samples from the compiled model
    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(compiled_hmm, (length(times), model_args_rest...))
        render_compiled_hmm_trace(scene, start, stop, maybe_path, times, speed, noise, dist_slack, trace, ax)
    end
    savefig("compiled_hmm_prior_samples.png")

end


##################################
# particle filtering experiments #
##################################

function experiment()

    Random.seed!(0)

    # generate a path and precompute distances from the start
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)
    distances_from_start = compute_distances_from_start(path)

    # generate ground truth locations and observations
    model_args = (length(times), path, distances_from_start, times, speed, noise, dist_slack)
    trace = simulate(lightweight_hmm, model_args)
    assignment = get_assignment(trace)
    measured_xs = [assignment[(:x, i)] for i=1:length(times)]
    measured_ys = [assignment[(:y, i)] for i=1:length(times)]
    actual_dists = [assignment[(:dist, i)] for i=1:length(times)]

    num_particles_list = [1, 3, 7, 10, 30, 100, 300, 1000]
    num_reps = 10
    max_steps = length(times)
    verbose = false


    ## particle filtering in compiled hmm using extenral proposal ##

    println("compiled hmm (external proposal)..")

    # These are computed in a strange place, but are the same for every
    # step, so I put the code here. posterior_var_d is the variance
    # of the posterior on d' given x and y. posterior_covars is a vector 
    # of 2x2 covariance matrices, representing the covariance of x and y
    # when d' has been marginalized out.
    posterior_var_d = dist_slack^2 * noise^2 / (dist_slack^2 + noise^2)
    posterior_covars = Vector{Matrix{Float64}}(undef, length(distances_from_start))
    for i = 2:length(distances_from_start)
        dd = distances_from_start[i] - distances_from_start[i-1]
        dx = path.points[i].x - path.points[i-1].x 
        dy = path.points[i].y - path.points[i-1].y
        posterior_covars[i] = [noise 0; 0 noise] .+ (dist_slack^2/dd^2 .* [dx^2 dx*dy; dy*dx dy^2])
    end

    function get_compiled_hmm_init()
        observations = DynamicAssignment()
        observations[1 => :x] = measured_xs[1]
        observations[1 => :y] = measured_ys[1]
        proposal_args = (1, (speed, times[1], dist_slack))
        return (observations, proposal_args)
    end

    function get_compiled_hmm_step(step::Int, trace)
        @assert step > 1
        observations = DynamicAssignment()
        observations[step => :x] = measured_xs[step]
        observations[step => :y] = measured_ys[step]
        prev_dist = get_assignment(trace)[step-1 => :dist]
        proposal_args = (step, (prev_dist, speed, times[step] - times[step-1], dist_slack))
        return (observations, proposal_args)
    end


    function get_fancy_compiled_hmm_init()
        observations = DynamicAssignment()
        observations[1 => :x] = measured_xs[1]
        observations[1 => :y] = measured_ys[1]
        proposal_args = (1, (times[1], Float64(0.0), noise, Point(measured_xs[1], measured_ys[1]), posterior_var_d, posterior_covars, path, distances_from_start, speed, dist_slack))
        return (observations, proposal_args)
    end

    function get_fancy_compiled_hmm_step(step::Int, trace)
        @assert step > 1
        observations = DynamicAssignment()
        observations[step => :x] = measured_xs[step]
        observations[step => :y] = measured_ys[step]
        prev_dist = get_assignment(trace)[step-1 => :dist]
        dt = step==1 ? times[step] : times[step] - times[step-1]
        proposal_args = (step, (dt, prev_dist, noise, Point(measured_xs[step], measured_ys[step]), posterior_var_d, posterior_covars,  path, distances_from_start, speed, dist_slack))
        return (observations, proposal_args)
    end

    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    results_compiled_external = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(compiled_hmm, model_args_rest, max_steps,
                                               num_particles, ess_threshold,
                                               get_fancy_compiled_hmm_init,
                                               get_fancy_compiled_hmm_step,
                                               compiled_fancy_proposal,
                                               compiled_fancy_proposal; verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_compiled_external[num_particles] = (lmls, elapsed)
    end

 # particle filtering in lightweight hmm using external proposal ##

    println("lightweight hmm (external proposal)..")

    function get_fancy_lightweight_hmm_init()
        observations = DynamicAssignment()
        observations[(:x, 1)] = measured_xs[1]
        observations[(:y, 1)] = measured_ys[1]
        proposal_args = (1, Float64(0.0), noise, Point(measured_xs[1], measured_ys[1]), posterior_var_d, posterior_covars,  path, distances_from_start, speed, times, dist_slack)
        return (observations, proposal_args)
    end

    function get_fancy_lightweight_hmm_step(step::Int, trace)
        @assert step > 1
        observations = DynamicAssignment()
        observations[(:x, step)] = measured_xs[step]
        observations[(:y, step)] = measured_ys[step]
        prev_dist = get_assignment(trace)[(:dist, step-1)]
        proposal_args = (step, prev_dist, noise, Point(measured_xs[step], measured_ys[step]), posterior_var_d, posterior_covars,  path, distances_from_start, speed, times, dist_slack)
        return (observations, proposal_args)
    end


    model_args_rest = (path, distances_from_start, times, speed, noise, dist_slack)
    results_lightweight_external = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(lightweight_hmm, model_args_rest, max_steps,
                                               num_particles, ess_threshold,
                                               get_fancy_lightweight_hmm_init,
                                               get_fancy_lightweight_hmm_step,
                                               lightweight_fancy_step_proposal,
                                               lightweight_fancy_step_proposal;verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_lightweight_external[num_particles] = (lmls, elapsed)
    end


    # particle filtering in lightweight hmm using internal proposal ##
    println("lightweight hmm (internal proposal)..")

    function get_lightweight_hmm_obs(step::Int)
        observations = DynamicAssignment()
        observations[(:x, step)] = measured_xs[step]
        observations[(:y, step)] = measured_ys[step]
        return observations 
    end

    model_args_rest = (path, distances_from_start, times, speed, noise, dist_slack)
    results_lightweight = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(lightweight_hmm, model_args_rest, max_steps, 
                                               num_particles, ess_threshold,
                                               get_lightweight_hmm_obs; verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_lightweight[num_particles] = (lmls, elapsed)
    end


   

    ## particle filtering in compiled hmm using internal proposal ##

    println("compiled hmm (internal proposal)..")

    function get_compiled_hmm_obs(step::Int)
        assignment = DynamicAssignment()
        assignment[step => :x] = measured_xs[step]
        assignment[step => :y] = measured_ys[step]
        return assignment
    end

    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    results_compiled = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(compiled_hmm, model_args_rest, max_steps,
                                                     num_particles, ess_threshold,
                                                     get_compiled_hmm_obs; verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_compiled[num_particles] = (lmls, elapsed)
    end



    # plot results

    figure(figsize=(8, 8))

    median_elapsed = [median(results_lightweight[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_lightweight[num_particles][1]) for num_particles in num_particles_list]
    println("lightweight (default)")
    println(results_lightweight)
#    println(results_lightweight[100][2])
#    println(median_elapsed)
#    println(mean_lmls)
    plot(median_elapsed, mean_lmls, label="lightweight (default)", color="blue")

    median_elapsed = [median(results_compiled[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_compiled[num_particles][1]) for num_particles in num_particles_list]
    println("compiled (default)")
    println(results_compiled)
    # println(results_compiled[100][2])    
    # println(median_elapsed)
    # println(mean_lmls)
    plot(median_elapsed, mean_lmls, label="compiled (default)", color="red")

    median_elapsed = [median(results_lightweight_external[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_lightweight_external[num_particles][1]) for num_particles in num_particles_list]
    println("lightweight (custom)")
    println(results_lightweight_external)
    # println(results_lightweight_external[100][2])
    # println(median_elapsed)
    # println(mean_lmls)
    plot(median_elapsed, mean_lmls, label="lightweight (custom)", color="cyan")

    median_elapsed = [median(results_compiled_external[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_compiled_external[num_particles][1]) for num_particles in num_particles_list]
    println("compiled (custom)")
    println(results_compiled_external)
    # println(results_compiled_external[100][2])
    # println(median_elapsed)
    # println(mean_lmls)
    plot(median_elapsed, mean_lmls, label="compiled (custom)", color="orange")


    ax = gca()
    ax[:set_xscale]("log")
    legend()
    xlabel("runtime (sec.)")
    ylabel("log marginal likelihood est.")
    savefig("filtering_results.png")
end

# NOTE: you have to separate compiled gen function definitions from calling API
# methods on them with a top-level call to this function
Gen.load_generated_functions()

# show_prior_samples()

experiment()
