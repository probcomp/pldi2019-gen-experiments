import JSON
import Statistics: median, quantile, mean

function mean_iqr(arr)
    m = median(arr)
    l = quantile(arr, 0.25)
    u = quantile(arr, 0.75)
    @assert u >= m
    @assert m >= l
    mean([u - m, m - l])
end

function print_results(results)
    elapsed = results["elapsed_list"]
    println("elapsed: $(median(elapsed)), +/- $(mean_iqr(elapsed))")
end

println("Gen (Dynamic)")
gen_dynamic_results = JSON.parsefile("gen/gen_results.json")["model"]
print_results(gen_dynamic_results)

println("Gen (Static)")
gen_static_results = JSON.parsefile("gen/gen_results.json")["static-model"]
print_results(gen_static_results)

println("Turing")
turing_results = JSON.parsefile("turing/turing_results.json")
print_results(turing_results)
