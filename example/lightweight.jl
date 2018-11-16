using Gen
import Random

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

@gen function datum(x::Float64, params::Params)
    if @addr(bernoulli(params.prob_outlier), :z)
        (mu, std) = (0., OUTLIER_STD)
    else
        (mu, std) = (x * params.slope + params.intercept, params.noise)
    end
    return @addr(normal(mu, std), :y)
end

data = plate(datum)

@gen function model(xs::Vector{Float64})
    prob_outlier = @addr(uniform(0, 1), :prob_outlier)
    noise = @addr(gamma(1, 1), :noise)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 2), :intercept)
    params = Params(prob_outlier, slope, intercept, noise)
    @diff begin
        addrs = [:prob_outlier, :slope, :intercept, :noise]
        diffs = [@choicediff(addr) for addr in addrs]
        argdiff = any(map(isnodiff, diffs)) ? unknownargdiff : noargdiff            
    end
    n = length(xs)
    ys = @addr(data(xs, fill(params, n)), :data, argdiff)
    return ys
end


#############
# rendering #
#############

using PyPlot
const POINT_SIZE = 10

function render_dataset(x::Vector{Float64}, y::Vector{Float64}, xlim, ylim)
    ax = plt[:gca]()
    ax[:scatter](x, y, c="black", alpha=1., s=POINT_SIZE)
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function render_linreg(trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true, show_line=true, show_points=true)
    xs = get_call_record(trace).args[1]
    assignment = get_assignment(trace)
    ax = plt[:gca]()
    if show_line
        slope = assignment[:slope]
        intercept = assignment[:intercept]
        line_xs = [xlim[1], xlim[2]]
        line_ys = slope * line_xs .+ intercept
        plt[:plot](line_xs, line_ys, color="black", alpha=line_alpha)
        noise = assignment[:noise]
        plt[:fill_between](line_xs, line_ys .- 2*noise, line_ys .+ 2*noise, color="black", alpha=0.2)
    end

    # plot data points
    if show_points
        colors = Vector{String}(undef, length(xs))
        ys = Vector{Float64}(undef, length(xs))
        for i=1:length(xs)
            if show_color
                is_outlier = assignment[:data => i => :z]
                color = is_outlier ? "red" : "blue"
            else
                color = "black"
            end
            y = assignment[:data => i => :y]
            colors[i] = color
            ys[i] = y
        end
        ax[:scatter](xs, ys, c=colors, alpha=point_alpha, s=POINT_SIZE)
    end
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
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
    @addr(uniform(0, 1), :prob_outlier)
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

#################
# plot data set #
#################

const FIGSIZE=(2,2)

xlim = (-5, 5)
ylim = (-15, 15)
figure(figsize=FIGSIZE)
render_dataset(xs, ys, xlim, ylim)
tight_layout()
savefig("data.pdf")


##################
# run experiment #
##################

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

    for i=1:n

        # steps on the parameters
        trace = mh(model, joint_proposal, (), trace)

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

## precompilation ##

(init_trace, trace, elapsed, scores) = do_inference(10)
(init_trace2, trace2, elapsed2, scores2) = do_generic_inference(10)

## actual experiment ##
xlim = (-5, 5)
ylim = (-15, 15)

Random.seed!(2)

(init_trace, trace, _, _) = do_inference(100)

figure(figsize=FIGSIZE)
render_linreg(init_trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true)
tight_layout()
savefig("init1.pdf")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true)
tight_layout()
savefig("final1.pdf")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, show_points=true, show_color=false, show_line=false)
tight_layout()
savefig("data.png")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, show_points=false)
tight_layout()
savefig("final_line.png")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; point_alpha=1.0, show_line=false, show_color=true)
tight_layout()
savefig("final_points.png")

exit()

(init_trace, trace, _, _) = do_generic_inference(100)

figure(figsize=FIGSIZE)
render_linreg(init_trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true)
tight_layout()
savefig("init2.pdf")

figure(figsize=FIGSIZE)
render_linreg(trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true)
tight_layout()
savefig("final2.pdf")

# do replicates
elapsed1_list = []
scores1_list = []
elapsed2_list = []
scores2_list = []

for i=1:4
    println("algorihtm 1 replicate $i")
    (_, _, elapsed, scores) = do_inference(200)
    push!(elapsed1_list, elapsed)
    push!(scores1_list, scores)
end

for i=1:4
    println("algorihtm 2 replicate $i")
    (_, _, elapsed, scores) = do_generic_inference(400)
    push!(elapsed2_list, elapsed)
    push!(scores2_list, scores)
end

figure(figsize=(4, 3))
for (i, (elapsed, scores)) in enumerate(zip(elapsed1_list, scores1_list))
    plot(elapsed, scores, color="green", label = i == 1 ? "Inference Program 1" : "")
end
for (i, (elapsed, scores)) in enumerate(zip(elapsed2_list, scores2_list))
    plot(elapsed, scores, color="brown", label = i == 1 ? "Inference Program 2" : "")
end
gca()[:set_ylim]((-300, 0))
#gca()[:set_xlim]((0, 5))
legend(loc="lower right")
ylabel("Log Probability")
xlabel("Seconds")
tight_layout()
savefig("scores.pdf")
