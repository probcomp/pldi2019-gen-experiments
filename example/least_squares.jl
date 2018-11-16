import Random

function least_squares(xs::Vector{Float64}, ys::Vector{Float64})
    n = length(xs)
    @assert length(ys) == n

    # y (n, 1)
    # X (n, 2)
    # theta (2, 1)
    # y = X * theta

    X = hcat(xs, ones(n))
    @assert size(X) == (n, 2)

    theta = X \ ys
    slope = theta[1]
    intercept = theta[2]
    return (slope, intercept)
end


Random.seed!(1)

xs = collect(range(-5, stop=5, length=200))
ys = Float64[]
true_slope = -1
true_intercept = 2
for (i, x) in enumerate(xs)
    y = true_slope * x + true_intercept + randn() * 0.1
    push!(ys, y)
end

(slope, intercept) = least_squares(xs, ys)
println(slope)
println(intercept)
