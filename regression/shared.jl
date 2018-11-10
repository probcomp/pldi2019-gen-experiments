using Random
using Gen

############
# modeling #
############

struct Params
    prob_outlier::Float64
    inlier_std::Float64
    outlier_std::Float64
    slope::Float64
    intercept::Float64
end

#####################
# generate data set #
#####################

function generate_dataset(xmin::Float64=-5., xmax::Float64=5., N::Int=200)
    prob_outlier = 0.5
    true_inlier_noise = 0.5
    true_outlier_noise = 5.0
    true_slope = -1
    true_intercept = 2
    xs = collect(range(xmin, stop=xmax, length=N))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        if rand() < prob_outlier
            y = true_slope * x + true_intercept + randn() * true_inlier_noise
        else
            y = true_slope * x + true_intercept + randn() * true_outlier_noise
        end
        push!(ys, y)
    end
    return (xs, ys)
end

#######################
# numerical functions #
#######################

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end
