using Printf: @sprintf
import Random
import Distributions
using Statistics: median, mean
using JLD

include("../gen-planning/scenes.jl")
include("../gen-planning/path_planner.jl")

################
# Turing model #
################

using Turing

@model turing_model(x_obs, y_obs, path, distances_from_start, times, speed, noise, dist_slack) = begin
    steps = length(x_obs)
    locations = Vector{Point}(undef, steps)
    dists = Vector{Float64}(undef,steps)
    
    dists[1] ~ Normal(speed * times[1], dist_slack)
    init_location = walk_path(path, distances_from_start, dists[1])
    x_obs[1] ~ Normal(init_location.x, noise)
    y_obs[1] ~ Normal(init_location.y, noise)
    
    for t=2:steps
        dists[t] ~ Normal(dists[t-1] + speed * (times[t] - times[t-1]), dist_slack)
        locations[t] = walk_path(path, distances_from_start, dists[t])
        point = locations[t]
        x_obs[t] ~ Normal(point.x, noise)
        y_obs[t] ~ Normal(point.y, noise)
    end

    return locations
end

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

##################################
# particle filtering experiments #
##################################

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

function estimate_marginal_likelihood(model, alg)
    # copied from https://github.com/TuringLang/Turing.jl/blob/master/src/samplers/smc.jl#L72
    spl = Turing.Sampler(alg)

    particles = Turing.ParticleContainer{Turing.Trace}(model)
    push!(particles, spl.alg.n_particles, spl, Turing.VarInfo())

    while Turing.consume(particles) != Val{:done}
        ess = Turing.effectiveSampleSize(particles)
        if ess <= spl.alg.resampler_threshold * length(particles)
            Turing.resample!(particles,spl.alg.resampler)
        end
    end

    return particles.logE
end

function turing_do_particle_filtering(params::Params,
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
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            alg = SMC(num_particles, Turing.resampleMultinomial, 0.5, Set(), 0)
            model = turing_model(measured_xs, measured_ys, path,
                               precomputed.distances_from_start, times,
                               speed, noise, dist_slack)
            lml = estimate_marginal_likelihood(model, alg)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml, elapsed: $(elapsed[rep])")
            lmls[rep] = lml
        end
        results[num_particles] = (lmls, elapsed)
    end
    return results
end

function experiment()

    Random.seed!(0)

    path = Path(Point(0.1, 0.1), Point(0.5, 0.5), Point[Point(0.1, 0.1), Point(0.0773627, 0.146073), Point(0.167036, 0.655448), Point(0.168662, 0.649074), Point(0.156116, 0.752046), Point(0.104823, 0.838075), Point(0.196407, 0.873581), Point(0.390309, 0.988468), Point(0.408272, 0.91336), Point(0.5, 0.5)])
measured_xs = [0.0890091, 0.105928, 0.158508, 0.11927, 0.0674466, 0.0920541, 0.0866197, 0.0828504, 0.0718467, 0.13625, 0.0949714, 0.0903477, 0.0994438, 0.073613, 0.0795392, 0.0993675, 0.111972, 0.0725639, 0.146364, 0.115776]
measured_ys = [0.25469, 0.506397, 0.452324, 0.377185, 0.23247, 0.110536, 0.11206, 0.115172, 0.170976, 0.0726151, 0.0763815, 0.0888771, 0.0683795, 0.0929964, 0.275081, 0.383367, 0.335842, 0.181095, 0.478705, 0.664434]

    # precomputation
    params = Params(times, speed, dist_slack, noise, path)
    precomputed = PrecomputedPathData(params)

    # parameters for particle filtering
    num_particles_list = [100000]#[1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300]
    num_reps = 10

    # experiments with compiled model
    results_turing = turing_do_particle_filtering(params,
            measured_xs, measured_ys, num_particles_list, num_reps)

    save("results_turing.jld", "results_turing", results_turing)

end

experiment()
