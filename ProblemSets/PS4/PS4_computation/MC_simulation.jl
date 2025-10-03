# This Julia script simulates a sufficiently long path from the Markov chain.
using Plots, Distributions, Random, LinearAlgebra
cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS4/PS4_computation")
include("Rouwenhorst_approx.jl")

function stationary_distribution(P)
    A = transpose(P) - I 
    A = vcat(A, ones(1, size(P, 1)))
    b = vcat(zeros(size(P, 1)), 1.0) # adding a summation constraint
    pi = A \ b
    return pi
end

all_models = collect(Base.Iterators.product([2, 5, 10, 20], [0.95, 0.99, 0.999]))

function simulate_MC(y0, states, P, T)
    state_idx = zeros(Int, T)
    state_idx[1] = argmin(abs.(states .- y0))

    for t in 2:T
        v = rand(Uniform(0,1))
        cdf = cumsum(@view P[state_idx[t-1], :])
        idx = searchsortedfirst(cdf, v)
        state_idx[t] = idx
    end
    return states[state_idx]
end

T = 100000
y0 = 0.0

for rho in [0.95, 0.99, 0.999]
    for num in [2, 5, 10, 20]
        ρ = rho
        N = num
        P, states = rouwenhorst(ρ, N; σ_z = 0.02/sqrt(1-ρ^2))
        println("working on :", "N = $N, ρ = $ρ")

        # Compute theoretical stationary distribution
        π_theoretic = stationary_distribution(P)
        y_path = simulate_MC(y0, states, P, T)
        y_path = y_path[Int(0.1*T):T]

        y_freq = [count(==(v), y_path) for v in states]
        y_frac = y_freq ./ sum(y_freq)

        # Plot the interpolated lines
        var = 0.02^2 / (1 - ρ^2)
        x = range(minimum(states), stop=maximum(states), length=2000)
        pdf_theoretical = pdf.(Normal(0, sqrt(var)), x)

        plot(x, pdf_theoretical, label = "Theoretical stationary distribution of AR(1)", 
            xlabel = "y", ylabel = "Density", lw = 3, color = :red, alpha = 0.5)

        # Fix bar plot: Ensure states and distributions align
        bar!(states, π_theoretic ./ (states[2] - states[1]), normalize = true, 
            label = "Stationary distribution of Markov chain", alpha = 0.5, 
            color = :green, linecolor=:transparent)

        bar!(states, y_frac ./ (states[2] - states[1]), normalize = true, 
            label = "Simulated distribution (N=$N, ρ=$ρ)", fillalpha=0, 
            linecolor=:blue, lw=2)
        plot!(legend=:outerbottom)
        # Save the plot
        savefig("images/MC_N$(N)_rho$(Int(ρ*1000))_theoretical_simulated.png")
    end
end


ρ = 0.99
N = 20
P, states = rouwenhorst(ρ, N; σ_z = 0.02/sqrt(1-ρ^2))
println("working on :", "N = $N, ρ = $ρ")

# Compute theoretical stationary distribution
π_theoretic = stationary_distribution(P)
y_path = simulate_MC(y0, states, P, T)
y_path = y_path[Int(0.1*T):T]

y_freq = [count(==(v), y_path) for v in states]
y_frac = y_freq ./ sum(y_freq)

# Plot the interpolated lines
var = 0.02^2 / (1 - ρ^2)
x = range(minimum(states), stop=maximum(states), length=2000)
pdf_theoretical = pdf.(Normal(0, sqrt(var)), x)

plot(x, pdf_theoretical, label = "Theoretical stationary distribution of AR(1)", 
    xlabel = "y", ylabel = "Density", lw = 3, color = :red, alpha = 0.5)

# Fix bar plot: Ensure states and distributions align
bar!(states, π_theoretic ./ (states[2] - states[1]), normalize = true, 
    label = "Stationary distribution of Markov chain", alpha = 0.5, 
    color = :green, linecolor=:transparent)

bar!(states, y_frac ./ (states[2] - states[1]), normalize = true, 
    label = "Simulated distribution (N=$N, ρ=$ρ)", fillalpha=0, 
    linecolor=:blue, lw=2)
plot!(legend=:outerbottom)
# Save the plot
savefig("images/MC_N$(N)_rho$(Int(ρ*1000))_theoretical_simulated.png")