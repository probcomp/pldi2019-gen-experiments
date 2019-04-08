include("../scenes.jl")
include("../path_planner.jl")
include("model.jl")

using Printf: @sprintf
import Random
using Statistics: median, std

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

@gen function stop_proposal(prev_trace)
    @trace(uniform(0, 1), :stop_x)
    @trace(uniform(0, 1), :stop_y)
end

@gen function speed_proposal(prev_trace)
    @trace(uniform(0, 1), :speed)
end

@gen function noise_proposal(prev_trace)
    @trace(uniform(0, 1), :noise)
end

function inference(model, measurements::Vector{Point}, start::Point, iters::Int)
    t = length(measurements)

    constraints = choicemap()
    for (i, pt) in enumerate(measurements)
        constraints[:measurements => i => :x] = pt.x
        constraints[:measurements => i => :y] = pt.y
    end
   constraints[:start_x] = start.x
   constraints[:start_y] = start.y

    (trace, _) = generate(model, (scene, times[1:t]), constraints)

    for iter=1:iters
        trace, = mh(trace, stop_proposal, ())
        trace, = mh(trace, speed_proposal, ())
        trace, = mh(trace, noise_proposal, ())
    end

    return trace
end

import JSON
function write_json_results(results, fname::AbstractString)
    open(fname,"w") do f
        JSON.print(f, results, 4)
    end
end

function show_paths(start, dest, speed, noise)
    Random.seed!(0)
    figure(figsize=(32, 32))
    constraints = choicemap((:start_x, start.x), (:start_y, start.y), (:stop_x, dest.x), (:stop_y, dest.y), (:speed, speed), (:noise, noise))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace, = generate(model, (scene, times), constraints)
        render(scene, trace, ax)
    end
    savefig("demo.png")
end

function experiment(reps::Int, iters::Int)

    # speed near 0.05, noise near 0.02
    measurements = Point[Point(0.0982709, 0.106993), Point(0.0994289, 0.181833), Point(0.134535, 0.219951), Point(0.137926, 0.256249), Point(0.137359, 0.296606), Point(0.0975037, 0.373101), Point(0.140863, 0.403996), Point(0.133527, 0.46508), Point(0.142269, 0.515338), Point(0.107248, 0.555732)]

    println(measurements)

    Random.seed!(1)

    start = Point(0.1, 0.1)

    results = Dict()
    T = length(measurements)
    for (model, model_name) in [
            (static_model, "static-model"),
            (static_model_no_cache, "static-model-no-cache"),
            (model, "model")]
        println(model_name)
        traces = []
        elapsed_list = []
        for i=1:reps
            println(i)
            start_time = time_ns()
            trace = inference(model, measurements[1:T], start, iters)
            elapsed = Int(time_ns() - start_time) / 1e9
            println("speed: $(trace[:speed]), noise: $(trace[:noise])")
            push!(elapsed_list, elapsed)
            push!(traces, trace)
        end
        median_elapsed = median(elapsed_list)
        std_elapsed = std(elapsed_list)
        results[model_name] = Dict("elapsed_list" => elapsed_list, "median_elapsed" => median_elapsed, "std_elapsed" => std_elapsed)
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
        fname = "inferred_$model_name.pdf"
        savefig(fname)
    end
    
    write_json_results(results, "gen_results.json")

end

load_generated_functions()

# do a run to force compilation 
println("initial run..")
experiment(50, 100)

# do a final run
println("final run..")
experiment(50, 1000)
