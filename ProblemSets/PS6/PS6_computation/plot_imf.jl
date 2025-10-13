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