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
        cdf = cumsum(@view P[state_idx[t-1], :])
        idx = searchsortedfirst(cdf, v)
        state_idx[t] = idx
    end
    return states[state_idx]
end

T = 2000
y0 = 0.0

y_path = []
for (N, ρ) in all_models
    P, states = rouwenhorst(ρ, N)
    y_path = simulate_MC(y0, states, P, T)

    # Compute auto-correlation 
    corr = cor(y_path[1:end-1], y_path[2:end])

    plot((T-1200):T, y_path[end-1200:end], label = "N = $N, ρ = $ρ", xlabel = "Time", ylabel = "y", legend = false)
    annotate!((T-1199), mean(y_path[end-1200:end]), text("Auto-corr: $(round(corr, digits=4))", :left, 10))
    savefig("images/MC_N$(N)_rho$(Int(ρ*1000)).png")
end
