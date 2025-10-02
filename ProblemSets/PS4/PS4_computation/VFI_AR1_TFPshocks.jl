using Plots, Distributions, Random, Parameters, UnPack

cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS4/PS4_computation")
include("Rouwenhorst_approx.jl")
include("MC_sim_func.jl")
include("VFI_function.jl")
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
K = 200
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

# Baseline case with fixed TFP 



#--------------------
# Baseline Case
#--------------------

vf, k_policy, c_policy = VFI_stochastic(model_params, computational_params)
vf_zmin_baseline = reshape(vf, N, K)[1, :]
vf_zmean_baseline = reshape(vf, N, K)[Int((N+1)/2), :]
vf_zmax_baseline = reshape(vf, N, K)[end, :]

kpr_zmin_baseline = reshape(k_policy, N, K)[1, :]
kpr_zmean_baseline = reshape(k_policy, N, K)[Int((N+1)/2), :]
kpr_zmax_baseline = reshape(k_policy, N, K)[end, :]

Δk_zmin_baseline = kpr_zmin_baseline .- k_vec
Δk_zmean_baseline = kpr_zmean_baseline .- k_vec
Δk_zmax_baseline = kpr_zmax_baseline .- k_vec

c_zmin_baseline = reshape(c_policy, N, K)[1, :]
c_zmean_baseline = reshape(c_policy, N, K)[Int((N+1)/2), :]
c_zmax_baseline = reshape(c_policy, N, K)[end, :]

#--------------------
# Deterministic Case
#--------------------
model_params = (
    γ = 2,
    δ = 0.05,
    β = 0.98,
    α = 1/3,
    A = 1
)

comp_params = (
    I = 200,
    max_iter = 1500
)

_, vf_deter, k_policy_deter, c_policy_deter = VFI(model_params, comp_params)
Δk_deter = k_policy_deter .- k_vec


# Visualize value functions for v(k, z_min) and v(k, z_max)
plot(k_vec, vf_zmin_baseline, 
    xlabel = "k", ylabel = "v(k,z)", 
    label = "Min TFP = $z_min", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, vf_zmax_baseline, 
    xlabel = "k", ylabel = "v(k,z)", 
    label = "Max TFP = $z_max", lw = 3, color = :red, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_valuefunc_min_max.png")


plot(k_vec, vf_deter, 
    xlabel = "k", ylabel = "v(k,z)", 
    label = "Deterministic", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, vf_zmean_baseline, 
    xlabel = "k", ylabel = "v(k,z)", 
    label = "Stochastic under zero TFP", lw = 3, color = :red, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_valuefunc_deterministic.png")



# Visualize policy function for kpr(k, z_min) and kpr(k, z_max)
k_policy_zmin = reshape(k_policy, N, K)[1, :]
k_policy_zmax = reshape(k_policy, N, K)[end, :]
z_min = round(z_vec[1], digits=3)
z_max = round(z_vec[end], digits=3)

plot(k_vec, k_policy_zmin, 
    xlabel = "k", ylabel = "g(k,z)", 
    label = "Min TFP = $z_min", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, k_policy_zmax, 
    xlabel = "k", ylabel = "g(k,z)", 
    label = "Max TFP = $z_max", lw = 3, color = :red, alpha = 0.9)
plot!(k_vec, k_vec, 
    label = "45 degree line", lw = 2, color = :black, alpha = 0.5)
savefig("VFI_AR1_TFPshocks_policyfunc_min_max.png")


plot(k_vec, k_policy_deter, 
    xlabel = "k", ylabel = "g(k)", 
    label = "Deterministic", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_zmean_baseline, 
    xlabel = "k", ylabel = "g(k,z)", 
    label = "Stochastic under zero TFP", lw = 3, color = :red, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_k_policy_deterministic.png")


## consumption policy 
plot(k_vec, c_zmin_baseline, 
    xlabel = "k", ylabel = "c(k,z)", 
    label = "Min TFP = $z_min", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, c_zmax_baseline, 
    xlabel = "k", ylabel = "c(k,z)", 
    label = "Max TFP = $z_max", lw = 3, color = :red, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_c_policy_min_max.png")


plot(k_vec, c_policy_deter, 
    xlabel = "k", ylabel = "c(k)", 
    label = "Deterministic", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, c_zmean_baseline, 
    xlabel = "k", ylabel = "c(k,z)", 
    label = "Stochastic under zero TFP", lw = 3, color = :green, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_c_policy_deterministic.png")


## change in capital 
plot(k_vec, Δk_zmin_baseline, 
    xlabel = "k", ylabel = "Δk(k,z) = g(k,z) - k", 
    label = "Min TFP = $z_min", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, Δk_zmax_baseline, 
    xlabel = "k", ylabel = "Δk(k,z) = g(k,z) - k", 
    label = "Max TFP = $z_max", lw = 3, color = :red, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_delta_k_policy_min_max.png")



plot(k_vec, Δk_deter, 
    xlabel = "k", ylabel = "g(k,z) - k", 
    label = "Deterministic", lw = 3, color = :blue, alpha = 0.9)
plot!(k_vec, Δk_zmean_baseline, 
    xlabel = "k", ylabel = "g(k,z) - k", 
    label = "Stochastic under zero TFP", lw = 3, color = :green, alpha = 0.9)
savefig("VFI_AR1_TFPshocks_delta_k_policy_deterministic.png")

# Impulse response 
T = 50
z0 = z_max 

z_path = simulate_MC(z0, z_vec, P, T)

