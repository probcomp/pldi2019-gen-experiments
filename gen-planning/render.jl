using PyPlot

const patches = PyPlot.matplotlib[:patches]

function render(polygon::Polygon, ax)
    num_sides = length(polygon.vertices)
    vertices = Matrix{Float64}(undef, num_sides, 2)
    for (i, pt) in enumerate(polygon.vertices)
        vertices[i,:] = [pt.x, pt.y]
    end
    poly = patches[:Polygon](vertices, true, facecolor="black")
    ax[:add_patch](poly)
end

render(wall::Wall, ax) = render(wall.poly, ax)
render(tree::Tree, ax) = render(tree.poly, ax)

function render(scene::Scene, ax)
    ax[:set_xlim]((scene.xmin, scene.xmax))
    ax[:set_ylim]((scene.ymin, scene.ymax))
    for obstacle in scene.obstacles
        render(obstacle, ax)
    end
end

function render(scene::Scene, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)
    (start, stop, speed, noise, maybe_path, locations) = get_call_record(trace).retval
    assignment = get_assignment(trace)
    render(scene, ax)
    sca(ax)
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    if show_measurements
        measured_xs = [assignment[i => :x] for i=1:length(locations)]
        measured_ys = [assignment[i => :y] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end