using Gen
import Random
using DataFrames
using CSV

#########
# model #
#########

struct Params
    prob_outlier::Float64
    slope::Float64
    intercept::Float64
    noise::Float64
end

const OUTLIER_STD = 10.

@gen (grad) function datum(x, prob_outlier, (grad)(slope), (grad)(intercept), noise)::Float64
    if @trace(bernoulli(prob_outlier), :z)
        (mu, std) = (0., OUTLIER_STD)
    else
        (mu, std) = (x * slope + intercept, noise)
    end
    return @trace(normal(mu, std), :y)
end

data = Map(datum)

@gen (grad, static) function model(xs::Vector{Float64})
    prob_outlier = @trace(uniform(0, 0.5), :prob_outlier)
    noise = @trace(gamma(1, 1), :noise)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    params = Params(prob_outlier, slope, intercept, noise)
    n = length(xs)
    ys = @trace(data(xs, fill(prob_outlier, n), fill(slope, n), fill(intercept, n), fill(noise, n)), :data)
    return ys
end
##########
# RANSAC #
##########

import StatsBase

struct RANSACParams

    # the number of random subsets to try
    iters::Int

    # the number of points to use to construct a hypothesis
    subset_size::Int

    # the error threshold below which a datum is considered an inlier
	# TODO: this should be tuned based on the inlier noise, no?
    eps::Float64
    
    function RANSACParams(iters, subset_size, eps)
        if iters < 1
            error("iters < 1")
        end
        new(iters, subset_size, eps)
    end
end


function ransac(xs::Vector{Float64}, ys::Vector{Float64}, params::RANSACParams)
    best_num_inliers::Int = -1
    best_slope::Float64 = NaN
    best_intercept::Float64 = NaN
    for i=1:params.iters

        # select a random subset of points
        rand_ind = StatsBase.sample(1:length(xs), params.subset_size, replace=false)
        subset_xs = xs[rand_ind]
        subset_ys = ys[rand_ind]
        
        # estimate slope and intercept using least squares
        A = hcat(subset_xs, ones(length(subset_xs)))
        slope, intercept = A\subset_ys
        
        ypred = intercept .+ slope * xs

        # count the number of inliers for this (slope, intercept) hypothesis
        inliers = abs.(ys .- ypred) .< params.eps
        num_inliers = sum(inliers)

        if num_inliers > best_num_inliers
            best_slope, best_intercept = slope, intercept
            best_num_inliers = num_inliers
        end
    end

    # return the hypothesis that resulted in the most inliers
    (best_slope, best_intercept)
end


#######################
# inference operators #
#######################

@gen function slope_proposal(prev)
    slope = get_choices(prev)[:slope]
    @trace(normal(slope, 0.2), :slope)
end

@gen function intercept_proposal(prev)
    intercept = get_choices(prev)[:intercept]
    @trace(normal(intercept, 0.2), :intercept)
end

@gen function noise_proposal(prev)
    noise = get_choices(prev)[:noise]
    @trace(gamma(1, 1), :noise)
end

@gen function prob_outlier_proposal(prev)
    @trace(uniform(0, 0.5), :prob_outlier)
end

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_choices(prev)[:data => i => :z]
    @trace(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

@gen function joint_proposal(prev)
    slope = get_choices(prev)[:slope]
    intercept = get_choices(prev)[:intercept]
    @trace(normal(slope, 0.2), :slope)
    @trace(normal(intercept, 0.2), :intercept)
    @trace(gamma(1, 1), :noise)
    @trace(uniform(0, 1), :prob_outlier)
end

#####################
# generate data set #
#####################

Random.seed!(1)

true_inlier_noise = 0.5
true_outlier_noise = OUTLIER_STD
prob_outlier = 0.1
true_slope = -1
true_intercept = 2
xs = collect(range(-5, stop=5, length=50))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = 0. + randn() * true_outlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_inlier_noise 
    end
    push!(ys, y)
end
ys[end-3] = 14
ys[end-5] = 13


######################
# inference programs #
######################

function print_trace(trace)
    score = get_score(trace)
    assignment = get_choices(trace)
    prob_outlier = assignment[:prob_outlier]
    slope = assignment[:slope]
    intercept = assignment[:intercept]
    noise = assignment[:noise]
    outlier_std = OUTLIER_STD
    println("score: $score, prob_outlier: $prob_outlier, slope: $slope, intercept: $intercept, inlier_std: $noise, outlier_std: $outlier_std")
end

time_sec() = time_ns() / 1e9
elapsed_sec(start::Float64) = (time_ns() / 1e9) - start

function do_gradient_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_score(trace)
    print_trace(trace)
    init_trace = trace

    selection = Gen.select(:slope, :intercept)

    for i=1:n

        # steps on the parameters
        for j=1:5
            trace, = mh(trace, prob_outlier_proposal, ())
            trace, = mh(trace, noise_proposal, ())
            trace, = mala(trace, selection, 0.001)
        end

        # step on the outliers
        for j=1:length(xs)
            trace, = mh(trace, is_outlier_proposal, (j,))
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_score(trace)
    end
    return (init_trace, trace, elapsed, scores)

end

@gen function ransac_proposal(prev_trace, xs, ys)
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    @trace(normal(slope, 0.1), :slope)
    @trace(normal(intercept, 1.0), :intercept)
end

function do_ransac_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_score(trace)
    print_trace(trace)
    init_trace = trace

    slope_selection = Gen.select(:slope)
    intercept_selection = Gen.select(:intercept)
    selection = Gen.select(:slope, :intercept)

    for i=1:n

        trace, = mh(trace, ransac_proposal, (xs, ys))

        # steps on the parameters
        for j=1:5
            trace, = mh(trace, prob_outlier_proposal, ())
            trace, = mh(trace, noise_proposal, ())
            trace, = mala(trace, selection, 0.001)
        end

        # step on the outliers
        for j=1:length(xs)
            trace, = mh(trace, is_outlier_proposal, (j,))
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_score(trace)
    end
    return (init_trace, trace, elapsed, scores)

end


function do_generic_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_score(trace)
    print_trace(trace)
    init_trace = trace

    selection = Gen.select(:noise, :prob_outlier, :slope, :intercept)
    for i=1:n

        trace, = mh(trace, selection)

        # step on the outliers
        for j=1:length(xs)
            trace, = mh(trace, is_outlier_proposal, (j,))
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_score(trace)
    end
    return (init_trace, trace, elapsed, scores)

end

function do_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # least squares
    X = hcat(xs, ones(length(xs)))
    (slope, intercept) = X \ ys
    println("slope: $slope")
    println("intercept: $intercept")
    constraints[:slope] = slope
    constraints[:intercept] = intercept

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_score(trace)
    print_trace(trace)
    init_trace = trace

    for i=1:n

        # steps on the parameters
        for j=1:5
            trace, = mh(trace, prob_outlier_proposal, ())
            trace, = mh(trace, noise_proposal, ())
            trace, = mh(trace, slope_proposal, ())
            trace, = mh(trace, intercept_proposal, ())
        end

        # step on the outliers
        for j=1:length(xs)
            trace, = mh(trace, is_outlier_proposal, (j,))
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_score(trace)
    end
    return (init_trace, trace, elapsed, scores)
end

Gen.load_generated_functions()

# precompile
(init_trace, trace, elapsed1, scores1) = do_generic_inference(10) # prog 1 
(init_trace, trace, elapsed2, scores2) = do_ransac_inference(10)  # prog 3
(init_trace, trace, elapsed3, scores3) = do_gradient_inference(10) # prog 2


#########################################
# generate 'results of inference' plots #
#########################################

const FIGSIZE=(2,2)
const xlim = (-5, 5)
const ylim = (-15, 15)

Random.seed!(2)

(init_trace, trace, _, _) = do_inference(100)

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, show_points=true, show_color=false, show_line=false)
tight_layout()
savefig("data.pdf")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, show_points=false)
tight_layout()
savefig("final_line.pdf")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; point_alpha=1.0, show_line=false, show_color=true)
tight_layout()
savefig("final_points.pdf")


#######################
# generate score plot #
#######################

# do the experiment

Random.seed!(2)

println("generic inference..")
(init_trace, trace, elapsed1, scores1) = do_generic_inference(1000) # blue (prog 1)

println("ransac inference..")
(init_trace, trace, elapsed2, scores2) = do_ransac_inference(100)  # orange (prog 3)

println("gradient inference..")
(init_trace, trace, elapsed3, scores3) = do_gradient_inference(100) # green (prog 2)

# save the data in CSVs for backup purposes

df = DataFrame()
df[:elapsed] = elapsed1
df[:scores] = scores1
CSV.write("example-data-prog1.csv", df)

df = DataFrame()
df[:elapsed] = elapsed2
df[:scores] = scores2
CSV.write("example-data-prog3.csv", df)

df = DataFrame()
df[:elapsed] = elapsed3
df[:scores] = scores3
CSV.write("example-data-prog2.csv", df)
