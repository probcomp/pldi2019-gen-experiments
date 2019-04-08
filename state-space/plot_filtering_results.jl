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
    #for i=1:length(num_particles_list)
        #t = median_times[i]
        #l = mean_lmls[i]
        #num_particles = num_particles_list[i]
        #text(t, l, "$num_particles")
    #end
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
const venture_num_particles_list = [1, 3, 10, 30, 100, 300]
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

gold_standard = mean(gen_results_lightweight_unfold_custom_proposal["3000"]["lmls"])
println("gold_standard: $gold_standard")

# print the runtime to cross the accuracy threshold

threshold = 56
print_crossing_point(turing_num_particles_list, turing_results, threshold, "Turing")
print_crossing_point(anglican_num_particles_list, anglican_results, threshold, "Anglican")
print_crossing_point(gen_num_particles_list, gen_results_lightweight_unfold_default_proposal, threshold, "Gen (Default Proposal)")
print_crossing_point(gen_num_particles_list, gen_results_lightweight_unfold_custom_proposal, threshold, "Gen (Custom Proposal)")
print_crossing_point(venture_num_particles_list, venture_results, threshold, "Venture")

# plot time accuracy curve

figure(figsize=(2.75,2.5))
plot_results(gen_num_particles_list, gen_results_lightweight_unfold_custom_proposal, "Gen (custom)", "red", "--")
plot_results(gen_num_particles_list, gen_results_lightweight_unfold_default_proposal, "Gen (generic)", "red", "-")
plot_results(anglican_num_particles_list, anglican_results, "Anglican", "blue")
plot_results(turing_num_particles_list, turing_results, "Turing", "purple")
plot_results(venture_num_particles_list, venture_results, "Venture", "green")
legend(loc="lower right")
ylabel("Accuracy (LML estimate)")
xlabel("Median runtime (sec.)")
gca()[:set_xscale]("log")
xlim = gca().get_xlim()
plot(xlim, [threshold, threshold], "-", color="black")
ax = gca()
ax.set_ylim((30, 60))
tight_layout()
savefig("lml_estimates.pdf")
