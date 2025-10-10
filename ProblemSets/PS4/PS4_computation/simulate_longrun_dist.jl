using Plots, Distributions, Random, Parameters, UnPack

cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS4/PS4_computation")
include("Rouwenhorst_approx.jl")
include("MC_sim_func.jl")
include("VFI_function.jl")
include("VFI_stochastic.jl")

# helper function 
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

# Compute the steady-state capital level
δ = 0.05
β = 0.98
α = 1/3
ρ = 0.95
σ_w = 0.02
A = 1
k_ss = ((1 - β + β * δ) / (β * A * α))^(1 / (α - 1))

# grids of capital 
K = 600
k_min = 0.1 * k_ss
k_max = 2 * k_ss
k_vec = convert(Array{Float64}, range(k_min, stop=k_max, length=K))

# grids of productivity
N = 21
σ_z = (σ_w)/(sqrt(1 - (ρ^2)))
P, z_vec = rouwenhorst(ρ, N; σ_z)
@assert z_vec[Int((N+1)/2)] == 0.0 "Error: zero productivity should locate in z_vec"
z_min = round(z_vec[1], digits=3)
z_max = round(z_vec[end], digits=3)

model_params = (
    γ = 2,
    δ = 0.05,
    β = 0.98,
    α = 1/3,
    A = 1,
    ρ = 0.95,
    σ_w = 0.02,
    P = P, 
    z_vec = z_vec, 
    k_vec = k_vec
)

computational_params = (
    diff = 10, 
    tol = 1e-6, 
    max_iter = 2200, 
    damp = 0.5 
)

# Derive policy and value functions
vf_sol, k_policy = VFI_stochastic(model_params, computational_params)

k_policy_mat = reshape(k_policy, N, K)


# Simulate long-run distribution of capital and productivity
TT = 11000
P, z_vec = rouwenhorst(ρ, N; σ_z = 0.02/sqrt(1-ρ^2))
z_path = simulate_MC(0.2, z_vec, P, TT)
k_path = ones(TT)
k_path[1] = 0.5 * k_ss
c_path = ones(TT)
i_path = ones(TT)
Y_path = ones(TT)


σz_sqr = (model_params.σ_w^2)/(1 - (model_params.ρ^2))

for tt in 2:TT
    z_curr = z_path[tt-1]
    k_curr = k_path[tt-1]

    z_idx = argmin(abs.(z_curr .- z_vec))
    k_idx = argmin(abs.(k_curr .- k_vec))
    k_next = k_policy_mat[z_idx, k_idx]
    k_path[tt] = k_next 
    Y = A * exp(-0.5 * σz_sqr) * exp(z_curr) * (k_curr^α)
    Y_path[tt-1] = Y
    c_curr = Y + (1 - δ) * k_curr - k_next
    c_path[tt-1] = c_curr
    i_curr = k_next - (1 - δ) * k_curr
    i_path[tt-1] = i_curr
end

# truncate the first 1000 periods 
k_path = k_path[1001: end-1]
c_path = c_path[1001: end-1]
i_path = i_path[1001: end-1]
Y_path = Y_path[1001: end-1]



# Plot histograms 
plot(
    histogram(k_path, bins=30, label="Capital", color=:blue, alpha=0.5, linecolor=:transparent),
    histogram(c_path, bins=30, label="Consumption", color=:blue, alpha=0.5,  linecolor=:transparent),
    histogram(i_path, bins=30, label="Investment", color=:blue, alpha=0.5,  linecolor=:transparent),
    histogram(Y_path, bins=30, label="Output", color=:blue, alpha=0.5, linecolor=:transparent);
    layout=(2, 2),  # Arrange plots in a 2x2 grid
    size=(1000, 800)  # Set the overall figure size
)

savefig("images/longrun_distributions.png")






# Impulse response 
σz_sqr = (model_params.σ_w^2)/(1 - (model_params.ρ^2))
T = 50
z0 = z_max 
I = 200 
z_paths = zeros(T+1, I)
k_paths = zeros(T+1, I)
c_paths = zeros(T, I)
i_paths = zeros(T, I)

z_paths[1, :] .= z_max
k_paths[1, :] .= k_ss

for i in 1:I
    Random.seed!(i)
    z_paths[:, i] = simulate_MC(z0, z_vec, P, (T+1))
    for t in 2:(T+1)
        z_curr = z_paths[t-1, i]
        k_curr = k_paths[t-1, i]

        z_idx = argmin(abs.(z_curr .- z_vec))
        k_idx = argmin(abs.(k_curr .- k_vec))
        k_next = k_policy_mat[z_idx, k_idx]
        k_paths[t, i] = k_next

        k_prod = A * exp(-0.5 * σz_sqr) * exp(z_curr) * (k_curr^α)
        i_curr = k_next - (1 - δ) * k_curr
        i_paths[t-1, i] = i_curr

        c_curr = k_prod + (1 - δ) * k_curr - k_next
        c_paths[t-1, i] = c_curr
    end
    if mod(i, 10) == 0 
        println("Finished round $i")
    end
end

# Visualize average paths of k, z, c, i 
# Plot capital paths
plot()
for i in 1:I
    plot!(1:(T+1), z_paths[1:T+1, i], color=:orange, alpha=0.1, label=false, lw=1)  # Add each path with light color
end
plot!(1:(T+1), mean(z_paths, dims=2), color=:blue, lw=3, label="Mean Productivity")  # Add mean path with bold line
savefig("images/avg_productivity_shocks.png")


plot()
for i in 1:I
    plot!(1:(T+1), k_paths[1:T+1, i], color=:orange, alpha=0.1, label=false, lw=1)  # Add each path with light color
end
plot!(1:(T+1), mean(k_paths, dims=2), color=:blue, lw=3, label="Mean Capital")  # Add mean path with bold line
savefig("images/avg_capital_impulse_response.png")

# Plot consumption paths
plot()
for i in 1:I
    plot!(1:T, c_paths[1:T, i], color=:orange, alpha=0.1, label=false, lw=1)  # Add each path with light color
end
plot!(1:T, mean(c_paths, dims=2), color=:blue, lw=3, label="Mean Consumption")  # Add mean path with bold line
savefig("images/avg_consumption_impulse_response.png")

# Plot investment paths
plot()
for i in 1:I
    plot!(1:T, i_paths[1:T, i], color=:orange, alpha=0.1, label=false, lw=1)  # Add each path with light color
end
plot!(1:T, mean(i_paths, dims=2), color=:blue, lw=3, label="Mean Investment")  # Add mean path with bold line
savefig("images/avg_investment_impulse_response.png")
