cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

# γ = 1
δ = 1
β = 0.98
α = 1/3
A = 1 
I = 500

k_min = 0.04 
k_max = 20
k_vec = k_min:(k_max - k_min)/(I-1) :k_max

v_old = zeros(I)
k_policy = zeros(I)

iter = 0 
max_iter = 1000
tol = 1e-8
diff = 10 

while diff > tol && iter < max_iter 

    consump_mat = (1 - δ) .* k_vec + A * (k_vec .^ α) .-  k_vec'

    # select infeasible consumption
    c_max = (1 - δ) * k_max + A * (k_max ^ α)
    infeasible_idx = (consump_mat .< 0) .| (consump_mat .> c_max)
    consump_mat[infeasible_idx] .= 1e-12
    # compute utility mat 
    util_mat = log.(consump_mat)
    util_mat[infeasible_idx] .= -Inf

    # update VF 
    v_imp = maximum(util_mat .+ β .* v_old', dims = 2)
    k_policy = k_vec[getindex.(argmax(util_mat .+ β .* v_old', dims = 2), 2)]

    diff = maximum(abs.(v_imp .- v_old))
    v_old = copy(v_imp)

    iter = iter + 1
    if mod(iter, 50) == 0
        println("Diff in $iter is :", diff)
    end
end

# Compute k-bar, s.t. g(k̄) = k̄
k_ss = k_vec[argmin(abs.(k_vec .-  k_policy))]


## Solutions from Guess and Verify 
b = α / (1 - α * β)
a_term1 = (log(1 - α * β)) / (1 - α * β)
# a_term2 = (α * β * log((α * β)/ (1 - α * β))) / (1 - α * β)
a_term2 = (α * β * log((α * β)/ (1 - α * β))) / (1 - α * β)
a = (a_term1 + a_term2) / (1 - β)

kpr_analytic = (β * b * (k_vec .^ α)) / (1 + b * β)
v_analytic = a .+ b .* log.(k_vec)

# Plotting k_vec
plot(k_vec, v_old, 
    xlabel = "Current capital", ylabel = "Value function", label = "VFI-based solution",
    lw = 2, color = :blue, alpha = 0.9)

plot!(k_vec, v_analytic, label = "Guess-and-Verify", color = :red, alpha = 0.9)
savefig("Cass_Koopman_VFI_value_func_simplified.png")






plot(k_vec, k_policy, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "VFI-based solution", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, kpr_analytic, 
    label = "Guess-and-Verify", lw = 2, color = :red, alpha = 0.9)

savefig("Cass_Koopman_VFI_policy_func_simplified.png")



# Projection
using DataFrames, GLM, StatsModels

# Example data
df = DataFrame(v_old = v_old[:], log_k = log.(k_vec))

# Fit OLS with intercept (default)
model = lm(@formula(v_old ~ log_k), df)

# Coefficients
coef(model)

