using Plots, Distributions, Random, Parameters, UnPack

cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS4/PS4_computation")
include("Rouwenhorst_approx.jl")
include("VFI_stochastic.jl")


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
k_min = 0
k_max = 2 * k_ss
k_vec = convert(Array{Float64}, range(k_min, stop=k_max, length=K))

# grids of productivity
N = 21
σ_z = (σ_w)/(sqrt(1 - (ρ^2)))
P, z_vec = rouwenhorst(ρ, N; σ_z)
@assert z_vec[Int((N+1)/2)] == 0.0 "Error: zero productivity should locate in z_vec"

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

#--------------------
# Deterministic Case
#--------------------
determin_model = merge(model_params, (σ_w = 0.0,))
vf_sol, k_policy = VFI_stochastic(determin_model, computational_params)




# Visualize value functions for v(k, z_min) and v(k, z_max)
vf_zmin = reshape(vf_sol, N, K)[1, :]
vf_zmax = reshape(vf_sol, N, K)[end, :]
z_min = round(z_vec[1], digits=3)
z_max = round(z_vec[end], digits=3)

plot(k_vec, vf_zmin, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "z = $z_min", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, vf_zmax, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "z = $z_max", lw = 2, color = :black, alpha = 0.9)


# Visualize policy function for kpr(k, z_min) and kpr(k, z_max)
k_policy_zmin = reshape(k_policy, N, K)[1, :]
k_policy_zmax = reshape(k_policy, N, K)[end, :]
z_min = round(z_vec[1], digits=3)
z_max = round(z_vec[end], digits=3)

plot(k_vec, k_policy_zmin, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "z = $z_min", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, k_policy_zmax, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "z = $z_max", lw = 2, color = :black, alpha = 0.9)
plot!(k_vec, k_vec, 
    label = "45 degree line", lw = 2, color = :red, alpha = 0.9)



