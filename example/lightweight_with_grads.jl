include("model.jl")

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
        
        ypred = intercept + slope * xs

        # count the number of inliers for this (slope, intercept) hypothesis
        inliers = abs.(ys - ypred) .< params.eps
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
    slope = get_assignment(prev)[:slope]
    @addr(normal(slope, 0.2), :slope)
end

@gen function intercept_proposal(prev)
    intercept = get_assignment(prev)[:intercept]
    @addr(normal(intercept, 0.2), :intercept)
end

@gen function noise_proposal(prev)
    noise = get_assignment(prev)[:noise]
    @addr(gamma(1, 1), :noise)
end

@gen function prob_outlier_proposal(prev)
    @addr(uniform(0, 0.5), :prob_outlier)
end

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

@gen function joint_proposal(prev)
    slope = get_assignment(prev)[:slope]
    intercept = get_assignment(prev)[:intercept]
    @addr(normal(slope, 0.2), :slope)
    @addr(normal(intercept, 0.2), :intercept)
    @addr(gamma(1, 1), :noise)
    @addr(uniform(0, 1), :prob_outlier)
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
    score = get_call_record(trace).score
    assignment = get_assignment(trace)
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

    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_call_record(trace).score
    print_trace(trace)
    init_trace = trace

    selection = Gen.select(:slope, :intercept)

    for i=1:n

        # steps on the parameters
        for j=1:5
            trace = mh(model, prob_outlier_proposal, (), trace)
            trace = mh(model, noise_proposal, (), trace)
            trace = mala(model, selection, trace, 0.001)
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_call_record(trace).score
    end
    return (init_trace, trace, elapsed, scores)

end

@gen function ransac_proposal(prev_trace, xs, ys)
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    @addr(normal(slope, 0.1), :slope)
    @addr(normal(intercept, 1.0), :intercept)
end

function do_ransac_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_call_record(trace).score
    print_trace(trace)
    init_trace = trace

    slope_selection = Gen.select(:slope)
    intercept_selection = Gen.select(:intercept)
    selection = Gen.select(:slope, :intercept)

    for i=1:n

        trace = mh(model, ransac_proposal, (xs, ys), trace; verbose=true)

        # steps on the parameters
        for j=1:5
            trace = mh(model, prob_outlier_proposal, (), trace)
            trace = mh(model, noise_proposal, (), trace)
            trace = mala(model, selection, trace, 0.001)
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_call_record(trace).score
    end
    return (init_trace, trace, elapsed, scores)

end


function do_generic_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), constraints)
    elapsed[1] = elapsed_sec(start)
    scores[1] = get_call_record(trace).score
    print_trace(trace)
    init_trace = trace

    selection = Gen.select(:noise, :prob_outlier, :slope, :intercept)
    for i=1:n

        trace = mh(model, selection, trace; verbose=true)

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_call_record(trace).score
    end
    return (init_trace, trace, elapsed, scores)

end

function do_inference(n)

    scores = Vector{Float64}(undef, n+1)
    elapsed = Vector{Float64}(undef, n+1)
    start = time_sec()

    constraints = DynamicAssignment()
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
    scores[1] = get_call_record(trace).score
    print_trace(trace)
    init_trace = trace

    for i=1:n

        # steps on the parameters
        for j=1:5
            trace = mh(model, prob_outlier_proposal, (), trace)
            trace = mh(model, noise_proposal, (), trace)
            trace = mh(model, slope_proposal, (), trace)
            trace = mh(model, intercept_proposal, (), trace)
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        print_trace(trace)
        elapsed[i+1] = elapsed_sec(start)
        scores[i+1] = get_call_record(trace).score
    end
    return (init_trace, trace, elapsed, scores)
end

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

# make the plot

figure(figsize=(4,3))
plot(elapsed1[2:2:end], scores1[2:2:end], color="blue", label="Inference Program 1")
plot(elapsed3[2:end], scores3[2:end], color="green", label="Inference Program 2")
plot(elapsed2[2:end], scores2[2:end], color="orange", label="Inference Program 3")
legend(loc="lower right")
ylabel("Log Probability")
xlabel("seconds")
gca()[:set_xlim]((0, 8))
tight_layout()
savefig("scores.pdf")


