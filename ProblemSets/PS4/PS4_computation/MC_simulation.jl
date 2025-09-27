# This Julia script simulates a sufficiently long path from the Markov chain.
using Plots, Distributions, Random
cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS4/PS4_computation")
include("Rouwenhorst_approx.jl")

all_models = collect(Base.Iterators.product([2, 5, 10, 20], [0.95, 0.99, 0.999]))

function simulate_MC(y0, states, P, T)

    state_idx = zeros(Int, T)
    state_idx[1] = argmin(abs.(states .- y0))

    for t in 2:T 
        # Random.seed!(t)
        v = rand(Uniform(0,1))
        state_idx[t] = min((1+sum(v .> cumsum(P[state_idx[t-1], :]))), length(states))
    end
    return states[state_idx]
end

T = 2000
y0 = 0.0

P, states = rouwenhorst(0.99, 20)
MC_path = simulate_MC(y0, states, P, T)

# Plot the histogram
histogram(MC_path[1001:end], bins = 30, title = "N = 20, ρ = 0.99", xlabel = "y", ylabel = "Frequency", legend = false)
# savefig("MC_hist_N20_rho990.png")

# Plot the last 1000 periods
plot(MC_path[1001:end], title = "N = 20, ρ = 0.99", xlabel = "Time", ylabel = "y", legend = false)

savefig("MC_N20_rho990.png")



for (N, ρ) in all_models
    σ_ϵ = 0.1 * sqrt(1 - ρ^2)
    P, states = rouwenhorst(ρ, σ_ϵ, N)
    y_path = simulate_MC(y0, states, P, T)

    plot(y_path, title = "N = $N, ρ = $ρ", xlabel = "Time", ylabel = "y", legend = false)
    savefig("MC_N$(N)_rho$(Int(ρ*1000)).png")
end