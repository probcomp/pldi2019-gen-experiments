using PyPlot
figure(figsize=(4,3))
plot([0.00349718, 0.0135704, 0.0316278, 0.0442908, 0.17273, 0.756062, 2.91378, 12.2601], 
	 [-4597.25, -459.825, -125.271, 1.14434, 62.1089, 66.1503, 67.6632, 67.7598], 
	 color="blue", 
	 label="Flexible DSL (default proposal)")
plot([0.000581203, 0.00160663, 0.0036468, 0.00516911, 0.0155486, 0.0545059, 0.183904, 0.691962],
[-4229.88, -423.875, -41.0699, 18.042, 60.2004, 66.2776, 67.7152, 67.8496],
	 color="orange", 
	 label="Restricted DSL (default proposal)")
# plot([0.17273, 0.756062, 2.91378, 12.2601], 
# 	 [62.1089, 66.1503, 67.6632, 67.7598], 
# 	 color="blue", 
# 	 label="Flexible DSL (default MH)")
# plot([0.0155486, 0.0545059, 0.183904, 0.691962],
# [60.2004, 66.2776, 67.7152, 67.8496],
# 	 color="orange", 
# 	 label="Restricted DSL (default MH)")
plot([0.00549324, 0.024556, 0.0549917, 0.0865796, 0.221581, 0.906985, 3.33198, 13.7389],
	 [63.9462, 66.1265, 66.8775, 67.3621, 67.1433, 67.1784, 67.2648, 67.2582],
	 color="lightblue", 
	 label="Flexible DSL (custom proposal)")
plot([0.00147046, 0.00438749, 0.00980691, 0.0135419, 0.0400791, 0.16141, 0.500942, 1.90717],
	 [64.351, 64.3574, 66.484, 66.4423, 67.0977, 67.3731, 67.2362, 67.2046],
	 color="red", 
	 label="Restricted DSL (custom proposal)")

legend(loc="lower right")
ylabel("log probability")
xlabel("seconds")
gca()[:set_xscale]("log")
tight_layout()
savefig("scores.pdf")