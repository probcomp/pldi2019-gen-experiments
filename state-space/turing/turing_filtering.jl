using Turing
using Statistics: median, mean

include("../geometry.jl")

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
const measured_xs = [0.0896684, 0.148145, 0.123211, 0.11035, 0.148417, 0.185746, 0.175872, 0.178704, 0.150475, 0.175573, 0.150151, 0.172628, 0.121426, 0.222041, 0.155273, 0.164001, 0.136586, 0.0687045, 0.146904, 0.163813]
const measured_ys = [0.217256, 0.416599, 0.376985, 0.383586, 0.500322, 0.608227, 0.632844, 0.653351, 0.532425, 0.881112, 0.771766, 0.653384, 0.756946, 0.870473, 0.8697, 0.808217, 0.598147, 0.163257, 0.611928, 0.657514]

@model turing_model(x_obs, y_obs, path, distances_from_start, times, speed, noise, dist_slack) = begin
    steps = length(x_obs)
    
    # walk path
    locations = Vector{Point}(undef, steps)
    dists = TArray{Float64}(undef, steps)
    dists[1] ~ Normal(speed * times[1], dist_slack)
    locations[1] = walk_path(path, distances_from_start, dists[1])
    x_obs[1] ~ Normal(locations[1].x, noise)
    y_obs[1] ~ Normal(locations[1].y, noise)
    for t=2:steps
        dists[t] ~ Normal(dists[t-1] + speed * (times[t] - times[t-1]), dist_slack)
        locations[t] = walk_path(path, distances_from_start, dists[t])
        point = locations[t]
        x_obs[t] ~ Normal(point.x, noise)
        y_obs[t] ~ Normal(point.y, noise)
    end

    return locations
end

function turing_pf(num_particles)
    spl = Turing.Sampler(Turing.SMC(num_particles, Turing.resample_multinomial, 0.5, Set(), 0))
    particles = Turing.ParticleContainer{Turing.Trace}(turing_model(measured_xs, measured_ys, path, distances_from_start, times, speed, noise, dist_slack))
    push!(particles, spl.alg.n_particles, spl, Turing.VarInfo())
    while Libtask.consume(particles) != Val{:done}
      ess = Turing.effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        Turing.resample!(particles,spl.alg.resampler)
      end
    end
    println(particles.logE)
    return particles.logE
end

import Random
Random.seed!(1)

num_reps = 50
const num_particles_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300]
results = Dict()
for num_particles in num_particles_list
    ess_threshold = num_particles / 2
    elapsed = Vector{Float64}(undef, num_reps)
    lmls = Vector{Float64}(undef, num_reps)
    for rep=1:num_reps
        start = time_ns()
    
        # run the particle filter
        lml = turing_pf(num_particles)

        # record results
        elapsed[rep] = Int(time_ns() - start) / 1e9
        lmls[rep] = lml
    end
    results[num_particles] = Dict("lmls" => lmls, "elapsed" => elapsed)
    println(mean(lmls))
end

println(results)
import JSON

open("turing_results.json","w") do f
    JSON.print(f, results, 4)
end
