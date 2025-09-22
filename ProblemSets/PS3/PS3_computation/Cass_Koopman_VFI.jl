cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

γ = 2 
δ = 0.05
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

    # compute utility mat 
    util_mat = (consump_mat.^(1 -  γ) .- 1) ./ (1 - γ)
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

# Plotting k_vec
plot(k_vec, v_old, 
    xlabel = "Current capital", ylabel = "Value function", 
    legend = false, lw = 2, color = :blue, alpha = 0.9)
savefig("Cass_Koopman_VFI_value_func.png")

plot(k_vec, k_policy, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Policy function", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, k_vec, 
    label = "45 degree line", lw = 2, color = :red, alpha = 0.9)
plot!([k_ss], [k_ss], seriestype = :scatter,
    label = "Steady-State Capital", marker = (:circle, 8), color = :red)

savefig("Cass_Koopman_VFI_policy_func.png")

