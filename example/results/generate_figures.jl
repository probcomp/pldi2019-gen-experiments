include("model.jl")


####################
# generate figures #
####################



exit()

## precompilation ##

(init_trace, trace, elapsed, scores) = do_inference(10)
(init_trace2, trace2, elapsed2, scores2) = do_generic_inference(10)


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
