include("scenes.jl")
include("path_planner.jl")
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
const times = collect(range(0, stop=1, length=20))

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

function experiment()

    measurements = Point[Point(0.200671, 0.0610075), Point(0.0386265, 0.018819), Point(-0.0936057, 0.198287), Point(0.0406614, 0.305687), Point(-0.0144924, 0.188301), Point(-0.00687671, 0.279064), Point(0.134002, 0.409317), Point(0.138309, 0.269417), Point(0.0254159, 0.494116), Point(0.201053, 0.317992), Point(0.0584362, 0.235333), Point(0.201222, 0.573131), Point(0.19059, 0.46727), Point(0.16229, 0.403766), Point(0.38414, 0.47726), Point(0.200641, 0.480193), Point(0.213562, 0.491452), Point(0.234423, 0.548481), Point(0.1109, 0.81866), Point(0.0310508, 0.629564)]

    start = Point(0.1, 0.1)

    results = Dict()
    reps = 10
    T = 10
    for (model, model_name) in [
            (model, "model"),
            (static_model, "static-model"),
            (static_model_no_cache, "static-model-no-cache")]
        println(model_name)
        traces = []
        elapsed_list = []
        for i=1:reps
            println(i)
            start_time = time_ns()
            trace = inference(model, measurements[1:T], start, 1000)
            elapsed = Int(time_ns() - start_time) / 1e9
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
            show_path=false, show_noise=false, stop_alpha=0.2, path_alpha=0.2)
        end
        fname = "inferred_$model_name.pdf"
        savefig(fname)
    end
    
    write_json_results(results, "gen_results.json")

end

Gen.load_generated_functions()
experiment()
