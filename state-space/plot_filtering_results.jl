using Statistics: median, mean, std
using PyPlot
using JLD
using Printf: @sprintf

function print_runtimes(num_particles_list::Vector{Int}, results::Dict, label::String)
    median_times = [median(results[num_particles][2]) for num_particles in num_particles_list]
    stdev_times = [std(results[num_particles][2]) for num_particles in num_particles_list]
    for (num_particles, median_time, stdev_time) in zip(num_particles_list, median_times, stdev_times)
        str = @sprintf("%s, %d particles: %0.3f +/- %0.3f", label, num_particles, median_time, stdev_time)
        println(str)
    end
end

function plot_results(num_particles_list::Vector{Int}, results::Dict, label::String, color::String)
    median_times = [median(results[num_particles][2]) for num_particles in num_particles_list]
    stdev_times = [std(results[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results[num_particles][1]) for num_particles in num_particles_list]
    stdev_lmls = [std(results[num_particles][1]) for num_particles in num_particles_list]
    println("min: $(minimum(mean_lmls)))")
    println("max: $(maximum(mean_lmls)))")
    plot(median_times, mean_lmls, 
	    color=color,
	    label=label)
end

results = load("results.jld")
results_turing = load("../turing-planning/results_turing.jld")

# experiments with static model
results_static_default_proposal = results["results_static_default_proposal"]
results_static_custom_proposal = results["results_static_custom_proposal"]

# experiments with lightweight model (no unfold)
results_lightweight_default_proposal = results["results_lightweight_default_proposal"]
results_lightweight_custom_proposal = results["results_lightweight_custom_proposal"]

# experiments with unfold
results_lightweight_unfold_default_proposal = results["results_lightweight_unfold_default_proposal"]
results_lightweight_unfold_custom_proposal = results["results_lightweight_unfold_custom_proposal"]

# Turing.jl
results_turing = results_turing["results_turing"]

const num_particles_list_default = [20, 30, 50, 70, 100, 200, 300]
const num_particles_list_custom = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300]

#######################
# print runtime table #
#######################

print_runtimes([100], results_turing, "Turing")
print_runtimes([100], results_static_default_proposal, "Restricted DSL  + unfold (default proposal)")
print_runtimes([100], results_lightweight_default_proposal, "Flexible DSL (default proposal)")
print_runtimes([100], results_lightweight_unfold_default_proposal, "Flexible DSL + unfold (default proposal)")

# Turing, 100 particles: 0.306 +/- 0.153
# Restricted DSL  + unfold (default proposal), 100 particles: 0.013 +/- 0.002
# Flexible DSL (default proposal), 100 particles: 0.926 +/- 0.066
#Flexible DSL + unfold (default proposal), 100 particles: 0.078 +/- 0.007

##################
# generate plots #
##################

# plot of all data

figure(figsize=(4,2))
plot_results(num_particles_list_custom, results_static_custom_proposal, "Custom Proposal", "orange")
plot_results(num_particles_list_default, results_static_default_proposal, "Default Proposal", "blue")
legend(loc="lower right")
ylabel("Accuracy (LML estimate)")
xlabel("seconds")
gca()[:set_xscale]("log")
tight_layout()
savefig("lml_estimates.pdf")
