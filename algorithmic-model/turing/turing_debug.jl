using Turing
import Random
using PyPlot

@model simple_model(z1, z2, z3) = begin
    x ~ Uniform(0., 1.)
    y ~ Uniform(0., 1.)
    z1 ~ Normal(x + y, 0.001)
    z2 ~ Normal(x, 0.01)
    z3 ~ Normal(y, 0.01)
end

function infer1(z1, z2, z3, num_iters::Int)
    chain = sample(simple_model(z1, z2, z3), MH(num_iters, :x, :y))
    x = chain[:x].value[end]
    y = chain[:y].value[end]
    (x, y)
end

function infer2(z1, z2, z3, num_iters::Int)
    chain = sample(simple_model(z1, z2, z3), Gibbs(num_iters, MH(1, :x, :y)))
    x = chain[:x].value[end]
    y = chain[:y].value[end]
    (x, y)
end

function infer3(z1, z2, z3, num_iters::Int)
    chain = sample(simple_model(z1, z2, z3), Gibbs(num_iters, MH(1, :x), MH(1, :y)))
    x = chain[:x].value[end]
    y = chain[:y].value[end]
    (x, y)
end

function show_results(infer, z1, z2, z3, num_iters, replicates, fname)
    xs = Vector{Float64}(undef, replicates)
    ys = Vector{Float64}(undef, replicates)
    for i=1:replicates
        (x, y) = infer(z1, z2, z3, num_iters)
        xs[i] = x
        ys[i] = y
    end
    figure(figsize=(4,4))
    scatter(xs, ys, marker=".", color="black")
    ax = gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    savefig(fname)
end

show_results(infer1, 1.0, 0.75, 0.25, 1000, 500, "infer1.pdf")
show_results(infer2, 1.0, 0.75, 0.25, 1000, 500, "infer2.pdf")
show_results(infer3, 1.0, 0.75, 0.25, 1000, 500, "infer3.pdf")
