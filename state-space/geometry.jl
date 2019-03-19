struct Point
    x::Float64
    y::Float64
end

Base.:+(a::Point, b::Point) = Point(a.x + b.x, a.y + b.y)
Base.:*(a::Real, b::Point) = Point(b.x * a, b.y * a)

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    sqrt(dx * dx + dy * dy)
end

struct Path
    start::Point
    goal::Point
    points::Array{Point,1}
end

function compute_distances_from_start(path::Path)
    distances_from_start = Vector{Float64}(undef, length(path.points))
    distances_from_start[1] = 0.0
    for i=2:length(path.points)
        distances_from_start[i] = distances_from_start[i-1] + dist(path.points[i-1], path.points[i])
    end
    return distances_from_start
end

function walk_path(path::Path, distances_from_start::Vector{Float64}, dist::Float64)
    if dist <= 0.
        return path.points[1]
    end
    if dist >= distances_from_start[end]
        return path.points[end]
    end
    # dist > 0 and dist < dist-to-last-point
    path_point_index = 0
    for i=1:length(distances_from_start)
        path_point_index += 1
        if dist < distances_from_start[path_point_index]
            break
        end
    end
    @assert dist < distances_from_start[path_point_index]
    path_point_index -= 1
    @assert path_point_index > 0
    dist_from_path_point = dist - distances_from_start[path_point_index]
    dist_between_points = distances_from_start[path_point_index + 1] - distances_from_start[path_point_index]
    fraction_next = dist_from_path_point / dist_between_points
    point::Point = (fraction_next * path.points[path_point_index + 1]
           + (1. - fraction_next) * path.points[path_point_index])
    return point
end

