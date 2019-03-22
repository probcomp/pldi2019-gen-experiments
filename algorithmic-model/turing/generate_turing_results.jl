include("../scenes.jl")
include("../path_planner.jl")

using Turing
using Statistics: median, mean
import Random

@model model(scene, times, xs, ys) = begin

    start_x = 0.1
    start_y = 0.1
    start = Point(start_x, start_y)
    
    stop_x ~ Uniform(0, 1)
    stop_y ~ Uniform(0, 1)
    stop = Point(stop_x, stop_y)
    
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    
    speed ~ Uniform(0, 1)
    
    locations = get_locations(maybe_path, start, speed, times)
    
    noise ~ Uniform(0, 0.1)

    for (i, point) in enumerate(locations)
        xs[i] ~ Normal(point.x, noise)
        ys[i] ~ Normal(point.y, noise)
    end

    nothing
end

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
const times = collect(range(0, stop=1, length=10))
const start = Point(0.1, 0.1)

function inference(model, measurements::Vector{Point}, start::Point, iters::Int)
    xs = [pt.x for pt in measurements]
    ys = [pt.y for pt in measurements]
    chain = sample(model(scene, times, xs, ys), Gibbs(iters, MH(1, :noise), MH(1, :speed), MH(1, :stop_x, :stop_y)))
    #chain = sample(model(scene, times, xs, ys), Gibbs(iters, MH(1, :stop_x, :stop_y)))
    #chain = sample(model(scene, times, xs, ys), MH(iters, :stop_x, :stop_y))
    trace = Dict(
        :stop_x => chain[:stop_x].value[end],
        :stop_y => chain[:stop_y].value[end],
        :noise => chain[:noise].value[end],
        :speed => chain[:speed].value[end]
    )
    return trace # i.e. trace
end

function render(scene::Scene, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1., path_line_alpha=0.5)
    stop = Point(trace[:stop_x], trace[:stop_y])
    render(scene, ax)
    sca(ax)
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha, zorder=2)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha, zorder=2)
    end
end

import JSON
function write_json_results(results, fname::AbstractString)
    open(fname,"w") do f
        JSON.print(f, results, 4)
    end
end

function experiment()

    # speed near 0.05, noise near 0.02
    measurements = Point[Point(0.0982709, 0.106993), Point(0.0994289, 0.181833), Point(0.134535, 0.219951), Point(0.137926, 0.256249), Point(0.137359, 0.296606), Point(0.0975037, 0.373101), Point(0.140863, 0.403996), Point(0.133527, 0.46508), Point(0.142269, 0.515338), Point(0.107248, 0.555732)]

    println(measurements)

    Random.seed!(1)


    reps = 50
    T = 10
    traces = []
    elapsed_list = []
    for i=1:reps
        println(i)
        start_time = time_ns()
        trace = inference(model, measurements[1:T], start, 1000)
        elapsed = Int(time_ns() - start_time) / 1e9
        println("trace: $trace")
        push!(elapsed_list, elapsed)
        push!(traces, trace)
    end
    median_elapsed = median(elapsed_list)
    std_elapsed = std(elapsed_list)
    results = Dict("elapsed_list" => elapsed_list, "median_elapsed" => median_elapsed, "std_elapsed" => std_elapsed)
    println("elapsed_list: $elapsed_list")
    println("median: $median_elapsed, std: $std_elapsed")

    # render results
    figure(figsize=(4, 4))
    ax = gca()
    ax[:axes][:xaxis][:set_ticks]([])
    ax[:axes][:yaxis][:set_ticks]([])
    for (i, trace) in enumerate(traces)
        render(scene, trace, ax; show_measurements=i==1, show_start=i==1,
        show_path=true, show_noise=false, stop_alpha=0.2, path_alpha=0.0, path_line_alpha=0.5)
    end
    fname = "inferred_turing.pdf"
    savefig(fname)
    
    write_json_results(results, "turing_results.json")
end

experiment()
