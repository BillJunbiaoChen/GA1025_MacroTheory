using Distributions, Statistics

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
mass_1 = vcat((20:-0.5:1), (1:0.5:5).*0.2, (1:0.5:20))
cdf_1 = vcat(0, cumsum(mass_1 ./ sum(mass_1)))

mass_2 = vcat((1:0.5:20) .* 0.05, (1:0.5:5), (20:-0.5:1) .* 0.25)
cdf_2 = vcat(0, cumsum(mass_2 ./ sum(mass_2)))

mass_3 = vcat((20:-0.5:1) .* 0.5, (1:0.5:5).*0.2, (1:0.5:20))
cdf_3 = vcat(0, cumsum(mass_3 ./ sum(mass_3)))


# Visualize 
# Base plot
plot(
    1:length(cdf_1),
    cdf_1,
    color=:blue,
    ylabel="CDF",
    legend=false,
    titlefont=10, guidefont=8, tickfont=6
)


plot!(
    1:length(cdf_1),
    cdf_2,
    color=:red,
    ylabel="CDF",
    legend=false
)


plot!(
    1:length(cdf_1),
    cdf_3,
    color=:black,
    ylabel="CDF",
    legend=false
)

annotate!(40, 0.55, text("F1", 8, :blue, :left))
annotate!(40, 0.2, text("F2", 8, :red, :left))
annotate!(40, 0.35, text("F3", 8, :black, :left))
savefig("single_crossing_is_not_transitive.png")
