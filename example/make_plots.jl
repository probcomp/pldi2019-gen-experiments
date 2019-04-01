using CSV
using DataFrames

#############
# rendering #
#############

using PyCall
@pyimport matplotlib.pyplot as plt

const POINT_SIZE = 10

function render_dataset(x::Vector{Float64}, y::Vector{Float64}, xlim, ylim)
    ax = plt[:gca]()
    ax[:scatter](x, y, c="black", alpha=1., s=POINT_SIZE)
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function render_linreg(trace, xlim, ylim; line_alpha=1.0, point_alpha=1.0, show_color=true, show_line=true, show_points=true)
    xs = get_args(trace)[1]
    assignment = get_choices(trace)
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

# make the plot

df = DataFrame(CSV.File("example-data-prog1.csv"))
elapsed1, scores1 = (df[:elapsed], df[:scores])

# df = DataFrame(CSV.File("example-data-prog3.csv"))
# elapsed2, scores2 = (df[:elapsed], df[:scores])

df = DataFrame(CSV.File("example-data-prog2.csv"))
elapsed2, scores2 = (df[:elapsed], df[:scores])

plt.figure()
plt.plot(elapsed1[2:end], scores1[2:end], color="orange", label="Inference Program 1")
plt.plot(elapsed2[2:end], scores2[2:end], color="green", label="Inference Program 2")
# plt.plot(elapsed2[2:end], scores2[2:end], color="orange", label="Inference Program 3")
plt.legend(loc="lower right")
plt.ylabel("Log Probability")
plt.xlabel("Runtime (seconds)")
plt.gca()[:set_xlim]((0, 2))
fig = plt.gcf()
fig[:set_size_inches]((5, 2))
fig[:tight_layout](pad=0)
plt.savefig("scores.pdf")
