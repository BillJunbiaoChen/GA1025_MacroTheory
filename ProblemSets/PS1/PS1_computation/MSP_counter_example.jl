using Distributions, Statistics, Random, Plots

# Define the mean vector and covariance matrix
μ = [1.0, 0.0]                # Mean vector
Σ = [1.0 0.5; 0.5 1.0]        # Covariance matrix (symmetric and positive definite)

# Create a 2-dimensional normal distribution
# [X, Z] ∼ N(μ, Σ)
Random.seed!(123)  # Set seed for reproducibility
samples = rand(MvNormal(μ, Σ), 10000)

# Show the sample mean of X
println("EX : ", mean(samples[1, :]))
println("EZ : ", mean(samples[2, :]))

# Define Y as X + Z
println("EY : ", mean(samples[1, :] + samples[2, :]))

println("Var(X)    : ", var(samples[1, :]))
println("Var(Z)    : ", var(samples[2, :]))
println("Var(Y)   : ", var(samples[1, :] + samples[2, :]))
println("Cov (X, Z)    : ", cov(samples[1, :], samples[2, :]))



# Single Crossing is not transitive
# Counter-example
x_vec = (0:0.05:1)
cdf_1 = x_vec
cdf_2 = vcat(3.6 .* (x_vec[1:10]).^2, (0.9:0.01:1) )
cdf_3 = vcat(zeros(10), 0.75.* ones(8), ones(3))



# Visualize 
# Base plot
plot(
    x_vec,
    cdf_1,
    color=:blue,
    ylabel="CDF",
    legend=false,
    titlefont=10, guidefont=8, tickfont=6
)


plot!(
    x_vec,
    cdf_2,
    color=:red,
    ylabel="CDF",
    legend=false
)


plot!(
    x_vec,
    cdf_3,
    color=:black,
    ylabel="CDF",
    legend=false
)

annotate!(0.1, 0.18, text("F1", 8, :blue, :left))
annotate!(0.5, 0.95, text("F2", 8, :red, :left))
annotate!(0.6, 0.8, text("F3", 8, :black, :left))
savefig("single_crossing_is_not_transitive.png")
