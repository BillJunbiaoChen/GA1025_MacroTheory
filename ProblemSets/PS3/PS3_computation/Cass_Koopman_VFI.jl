cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

γ = 2 
δ = 0.05
β = 0.98
α = 1/3
A = 1 


k_ss = ((1 - β + β * δ)/(β * A * α))^(1/(α - 1))

# Question 1.3
k_min = 0.95 * round(k_ss, digits = 8)
k_max = 1.05 * round(k_ss, digits = 8)
I = 500
k_vec = k_min:(k_max - k_min)/(I-1) :k_max


# Question 1.4
# I = 800 
# k_min = 0.1 * round(k_ss, digits = 8)
# k_max = 2 * round(k_ss, digits = 8)
# k_vec = k_min:(k_max - k_min)/(I-1) :k_max


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

# Plotting k_vec
plot(k_vec, v_old, 
    xlabel = "Current capital", ylabel = "Value function", 
    legend = false, lw = 2, color = :blue, alpha = 0.9)


if k_min == 0.1 * round(k_ss, digits = 8)
    savefig("Cass_Koopman_VFI_value_func_robust.png")
end

if k_min == 0.95 * round(k_ss, digits = 8)
    savefig("Cass_Koopman_VFI_value_func.png")
end



plot(k_vec, k_policy, 
    xlabel = "Current capital", ylabel = "Next period capital", 
    label = "Policy function", lw = 2, color = :blue, alpha = 0.9)
plot!(k_vec, k_vec, 
    label = "45 degree line", lw = 2, color = :red, alpha = 0.9)
plot!([k_ss], [k_ss], seriestype = :scatter,
    label = "Steady-State Capital", marker = (:circle, 8), color = :red)

if k_min == 0.1 * round(k_ss, digits = 8)
    savefig("Cass_Koopman_VFI_policy_func_robust.png")
end

if k_min == 0.95 * round(k_ss, digits = 8)
    savefig("Cass_Koopman_VFI_policy_func.png")
end

