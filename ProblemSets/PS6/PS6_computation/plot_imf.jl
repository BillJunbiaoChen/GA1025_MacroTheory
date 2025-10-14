cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS6/PS6_computation")

using Plots 
β = 0.95
ρ = 0.5 
σ1 = 1
σ2 = 1

t_vec = 1:20
hb1_vec = -σ1 .* (1 .- (ρ.^t_vec)) ./ (1 - β * ρ)

plot(t_vec, hb1_vec, 
    xlabel = "Time", ylabel = "Debt's impulse response", 
    ylim = (-2,0), label = "Shock to the 1st component (Full info)", lw = 3, color = :blue, alpha = 0.9)

plot!(t_vec, -σ2 .* ones(length(t_vec)), 
    xlabel = "Time", ylabel = "Debt's impulse response", 
    ylim = (-2,0), label = "Shock to the 2nd component (Full info)", lw = 3, color = :red, alpha = 0.9)

# plot!(legend = :right)

savefig("../images/debt_impulse_response_functions.png") 


plot(t_vec,  ((1 - β) * σ1 / (1 - β * ρ)) .* ones(length(t_vec)), 
    xlabel = "Time", ylabel = "Consumption's impulse response", 
    ylim = (0, 0.3), label = "Shock to the 1st component (Full info)", lw = 3, color = :blue, alpha = 0.9)

plot!(t_vec, ((1 - β) * σ2 ).* ones(length(t_vec)), 
    xlabel = "Time", ylabel = "Consumption's impulse response", 
    ylim = (0, 0.3), label = "Shock to the 2nd component (Full info)", lw = 3, color = :red, alpha = 0.9)

plot!(legend = :right)

savefig("../images/consumption_impulse_response_functions.png") 

# Parameters
t_vec = 1:20
σ1 = 1.0
β = 0.95

# Evolution of hb as rho approaches 1
I = 30
hb_mat = ones(length(t_vec), I)  # Initialize `hb_mat`
rho_vec = 0.5:0.01:0.79          # Values of `rho`


for (i, rho) in enumerate(rho_vec)
    hb_mat[:, i] .= -σ1 .* (1 .- (rho .^ t_vec)) ./ (1 - β * rho)
end

# Generate colors using the color gradient
colors = cgrad(:viridis, size(hb_mat, 2))

# Base plot
plot(
    t_vec,
    hb_mat[:, 1],
    xlabel = "Time",
    ylabel = "Debt's impulse response",
    guidefontsize = 8,
    tickfontsize = 6,
    legend = false
)


for t in 2:size(hb_mat, 2)
    plot!(t_vec, hb_mat[:, t], color = colors[t], label = "") 
end

# Add annotations
x_anno = 10
y_anno_start = hb_mat[x_anno, 1]  # Corresponding y-coordinate
annotate!(x_anno, y_anno_start, text("ρ = $(round(rho_vec[1], digits=2))", 8, :blue, :left))
y_anno_end = hb_mat[x_anno, end]
annotate!(x_anno, y_anno_end * 1.001, text("ρ = $(round(rho_vec[end], digits=2))", 8, :blue, :left))

# Save the figure
savefig("../images/convergence_of_debt_impulse_response_functions.png")