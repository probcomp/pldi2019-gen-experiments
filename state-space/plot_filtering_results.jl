using Statistics: median, mean, std, quantile
using PyPlot
using Printf: @sprintf
import JSON

function mean_iqr(arr)
    m = median(arr)
    l = quantile(arr, 0.25)
    u = quantile(arr, 0.75)
    @assert u >= m
    @assert m >= l
    mean([u - m, m - l])
end

function plot_results(num_particles_list::Vector{Int}, results::Dict, label::String,
            color::String, linestyle="-")
    median_times = [median(results[string(num_particles)]["elapsed"]) for num_particles in num_particles_list]
    mean_lmls = [mean(results[string(num_particles)]["lmls"]) for num_particles in num_particles_list]
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
    elapsed = results[string(chosen_num_particles)]["elapsed"] * 1000 # sec. to ms.
    median_elapsed = median(elapsed)
    mean_iqr_elapsed = mean_iqr(elapsed)
    mean_lml = mean(results[string(chosen_num_particles)]["lmls"])
    str = @sprintf("%s, %s %d particles: %0.3fms (+/- %0.3f), mean lml: %0.3f", name, success ? "" : ">", chosen_num_particles,
            median_elapsed, mean_iqr_elapsed, mean_lml)
    println(str)
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

# NOTE: for Venture, there seems to be a bug.
# since the algorithm is the same as the Anglican algorithm (default proposal,
# resampling at every step), which reaches the threshold after 200 particles,
# we will use the time measurements for Venture with 100 particles, and state >
# X where X is that time.
venture_elapsed = venture_results["100"]["elapsed"] * 1000 # sec. to ms.
median_elapsed = median(venture_elapsed)
mean_iqr_elapsed = mean_iqr(venture_elapsed)
println("Venture: > $median_elapsed (+/- $mean_iqr_elapsed)")

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
