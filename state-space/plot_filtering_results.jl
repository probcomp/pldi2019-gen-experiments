using Statistics: median, mean, std
using PyPlot
using Printf: @sprintf
import JSON

function print_runtimes(num_particles_list::Vector{Int}, results::Dict, label::String)
    median_times = [median(results[num_particles][2]) for num_particles in num_particles_list]
    stdev_times = [std(results[num_particles][2]) for num_particles in num_particles_list]
    for (num_particles, median_time, stdev_time) in zip(num_particles_list, median_times, stdev_times)
        str = @sprintf("%s, %d particles: %0.3f +/- %0.3f", label, num_particles, median_time, stdev_time)
        println(str)
    end
end

function plot_results(num_particles_list::Vector{Int}, results::Dict, label::String,
            color::String, linestyle="-")
    println(keys(results))
    median_times = [median(results[string(num_particles)]["elapsed"]) for num_particles in num_particles_list]
    stdev_times = [std(results[string(num_particles)]["elapsed"]) for num_particles in num_particles_list]
    mean_lmls = [mean(results[string(num_particles)]["lmls"]) for num_particles in num_particles_list]
    stdev_lmls = [std(results[string(num_particles)]["lmls"]) for num_particles in num_particles_list]
    println("min: $(minimum(mean_lmls)))")
    println("max: $(maximum(mean_lmls)))")
    plot(median_times, mean_lmls, 
	    color=color,
	    label=label,
        linestyle=linestyle)
end

function print_crossing_point(num_particles_list::Vector{Int}, results::Dict, threshold::Real, name::String)
    success = false
    local chosen_num_particles::Int
    for num_particles in num_particles_list
        mean_lml = mean(results[string(num_particles)]["lmls"])
        if mean_lml > threshold
            chosen_num_particles = num_particles
            success = true
            break
        end
    end
    if !success
        chosen_num_particles = maximum(num_particles_list)
    end
    median_elapsed = median(results[string(chosen_num_particles)]["elapsed"])
    mean_lml = mean(results[string(chosen_num_particles)]["lmls"])
    std_lml = std(results[string(chosen_num_particles)]["lmls"])
    println("$name, $(success ? "" : ">") $chosen_num_particles particles, median elapsed: $median_elapsed, mean_lml: $mean_lml, +/-: $std_lml")
end


# load data 

const turing_num_particles_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000]
const gen_num_particles_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000]
const venture_num_particles_list = [1, 3, 10, 30]
const anglican_num_particles_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000]

anglican_results = JSON.parsefile("anglican/anglican-results.json")
venture_results = JSON.parsefile("venture/venture_results.json")
turing_results = JSON.parsefile("turing/turing_results.json")
gen_results_static_default_proposal = JSON.parsefile("gen/gen_results_static_default_proposal.json")
gen_results_lightweight_unfold_custom_proposal = JSON.parsefile("gen/gen_results_lightweight_unfold_custom_proposal.json")
gen_results_lightweight_unfold_default_proposal = JSON.parsefile("gen/gen_results_lightweight_unfold_default_proposal.json")
gen_results_lightweight_custom_proposal = JSON.parsefile("gen/gen_results_lightweight_custom_proposal.json")
gen_results_lightweight_default_proposal = JSON.parsefile("gen/gen_results_lightweight_default_proposal.json")
gen_results_static_custom_proposal = JSON.parsefile("gen/gen_results_static_custom_proposal.json")

# print the runtime to cross the accuracy threshold

threshold = 56
print_crossing_point(turing_num_particles_list, turing_results, threshold, "Turing")
print_crossing_point(anglican_num_particles_list, anglican_results, threshold, "Anglican")
print_crossing_point(gen_num_particles_list, gen_results_lightweight_unfold_default_proposal, threshold, "Gen (Default Proposal)")
print_crossing_point(gen_num_particles_list, gen_results_lightweight_unfold_custom_proposal, threshold, "Gen (Custom Proposal)")
print_crossing_point(venture_num_particles_list, venture_results, threshold, "Venture")

# plot time accuracy curve

figure(figsize=(8,4))
plot_results(anglican_num_particles_list, anglican_results, "Anglican", "blue")
plot_results(venture_num_particles_list, venture_results, "Venture", "green")
plot_results(turing_num_particles_list, turing_results, "Turing", "purple")
plot_results(gen_num_particles_list, gen_results_lightweight_unfold_default_proposal, "Gen (Default Proposal)", "red", "--")
plot_results(gen_num_particles_list, gen_results_lightweight_unfold_custom_proposal, "Gen (Custom Proposal)", "orange", "--")
legend(loc="lower right")
ylabel("Accuracy (LML estimate)")
xlabel("seconds")
gca()[:set_xscale]("log")
ax = gca()
ax.set_ylim((0, 60))
tight_layout()
savefig("lml_estimates.pdf")
