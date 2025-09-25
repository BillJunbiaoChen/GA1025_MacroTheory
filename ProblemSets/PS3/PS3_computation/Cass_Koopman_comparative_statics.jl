cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra, Parameters, UnPack

include("VFI_function.jl")


computational_parameters = (I = 800, max_iter = 1200)
model_param0 = (
    γ = 2,
    δ = 0.05,
    β = 0.98,
    α = 1/3,
    A = 1
)

# Question 1.5 
## baseline 
k_vec, v_sol_0, kpr_sol_0 = VFI(model_param0, computational_parameters)

## ------------------------------- ##
## Part 1: increased depreciation 
## ------------------------------- ##
model_param1 = (
    γ = 2,
    δ = 0.1,
    β = 0.98,
    α = 1/3,
    A = 1
)

k_vec, v_sol_1, kpr_sol_1 = VFI(model_param1, computational_parameters)


# Plotting k_vec
plot(k_vec, v_sol_0, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)

plot!(k_vec, v_sol_1, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Increased depreciation", lw = 2, color = :red, alpha = 0.9)

savefig("Cass_Koopman_VFI_value_func_inc_delta.png")


plot(k_vec, kpr_sol_0, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_sol_1, 
    label = "Increased depreciation", lw = 2, color = :red, alpha = 0.9)
savefig("Cass_Koopman_VFI_policy_func_inc_delta.png")


## ------------------------------- ##
## Part 2: More impatience
## ------------------------------- ##
model_param2 = (
    γ = 2,
    δ = 0.05,
    β = 0.95,
    α = 1/3,
    A = 1
)

k_vec, v_sol_2, kpr_sol_2 = VFI(model_param2, computational_parameters)


# Plotting k_vec
plot(k_vec, v_sol_0, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)

plot!(k_vec, v_sol_2, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "More impatience", lw = 2, color = :red, alpha = 0.9)

savefig("Cass_Koopman_VFI_value_func_dec_beta.png")


plot(k_vec, kpr_sol_0, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_sol_2, 
    label = "More impatience", lw = 2, color = :red, alpha = 0.9)
savefig("Cass_Koopman_VFI_policy_func_dec_beta.png")


## ------------------------------- ##
## Part 3: Higher captial share
## ------------------------------- ##
model_param3 = (
    γ = 2,
    δ = 0.05,
    β = 0.98,
    α = 1/2,
    A = 1
)

k_vec, v_sol_3, kpr_sol_3 = VFI(model_param3, computational_parameters)


# Plotting k_vec
plot(k_vec, v_sol_0, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)

plot!(k_vec, v_sol_3, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Higher capital share", lw = 2, color = :red, alpha = 0.9)

savefig("Cass_Koopman_VFI_value_func_inc_alpha.png")


plot(k_vec, kpr_sol_0, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_sol_3, 
    label = "Higher capital share", lw = 2, color = :red, alpha = 0.9)
savefig("Cass_Koopman_VFI_policy_func_inc_alpha.png")





## ------------------------------- ##
## Part 4: Lower IES
## ------------------------------- ##
model_param4 = (
    γ = 3,
    δ = 0.05,
    β = 0.98,
    α = 1/3,
    A = 1
)

k_vec, v_sol_4, kpr_sol_4 = VFI(model_param4, computational_parameters)


# Plotting k_vec
plot(k_vec, v_sol_0, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Baseline", lw = 2, color = :blue, alpha = 0.9)

plot!(k_vec, v_sol_4, 
    xlabel = "Current capital", ylabel = "Value function", 
    label = "Lower intertemporal elasticity of substitution", lw = 2, color = :red, alpha = 0.9)

savefig("Cass_Koopman_VFI_value_func_inc_gamma.png")


plot(k_vec, kpr_sol_0, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Baseline", lw = 1, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_sol_4, 
    label = "Lower intertemporal elasticity of substitution", lw = 1, color = :red, alpha = 0.9)
savefig("Cass_Koopman_VFI_policy_func_inc_gamma.png")



