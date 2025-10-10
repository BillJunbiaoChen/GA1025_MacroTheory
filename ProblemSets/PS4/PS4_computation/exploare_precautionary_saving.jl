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
K = 300
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

computational_params = (
    diff = 10, 
    tol = 1e-6, 
    max_iter = 2200, 
    damp = 0.5 
)

# 
k_mean_vec = ones(4)

for (i, gamma) in enumerate([2, 3, 4 , 5])
    model_params = (
        γ = gamma,
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

    vf_sol, k_policy = VFI_stochastic(model_params, computational_params)

    k_policy_mat = reshape(k_policy, N, K)


    # Simulate long-run distribution of capital and productivity
    TT = 11000
    P, z_vec = rouwenhorst(ρ, N; σ_z = 0.02/sqrt(1-ρ^2))
    z_path = simulate_MC(0.2, z_vec, P, TT)
    k_path = ones(TT)
    k_path[1] = 0.5 * k_ss

    σz_sqr = (model_params.σ_w^2)/(1 - (model_params.ρ^2))

    for tt in 2:TT
        z_curr = z_path[tt-1]
        k_curr = k_path[tt-1]

        z_idx = argmin(abs.(z_curr .- z_vec))
        k_idx = argmin(abs.(k_curr .- k_vec))
        k_next = k_policy_mat[z_idx, k_idx]
        k_path[tt] = k_next 
    end

    # truncate the first 1000 periods 
    k_path = k_path[1001: end-1]
    k_mean_vec[i] = mean(k_path)
end


# Plot histograms 
plot([2, 3, 4 , 5], k_mean_vec, label = "sample-mean", lw = 3, color = :red, alpha = 0.7)
plot!([2, 3, 4 , 5], ones(4) .* k_ss, label = "steady-state", lw = 3, color = :blue, alpha = 0.7)

savefig("images/precautionary_saving.png")

