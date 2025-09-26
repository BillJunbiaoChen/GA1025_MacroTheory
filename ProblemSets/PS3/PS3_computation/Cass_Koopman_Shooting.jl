cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

# Parameters
γ = 2 
δ = 0.05
β = 0.98
α = 1/3
A = 1 

# Compute the steady-state capital level
k_ss = ((1 - β + β * δ) / (β * A * α))^(1 / (α - 1))

# Compute the Transversality Condition (TVC)
function tvc_compute(kt, ktp, t)
    tvc_term1 = ((1 - δ) * kt + A * (kt ^ α) - ktp) ^ (-γ)
    tvc_term2 = 1 - δ + A * α * (kt ^ (α - 1))
    tvc = (β^t) * tvc_term1 * tvc_term2 * kt
    return tvc 
end

# Compute the capital evolution starting from k0 and k1
function k_end(k0, k1, T)
    k_vec = zeros(T + 1)
    k_vec[1] = k0
    k_vec[2] = k1

    tvc_vec = zeros(T)
    tvc_vec[1] = tvc_compute(k0, k1, 1) # t = 1
    
    for t in 3:(T + 1)
        term1 = (1 - δ) * k_vec[t-1] + A * (k_vec[t-1]^α)
        term2 = (β^(1 / γ)) * (1 - δ + A * α * k_vec[t-1]^(α-1))^(1 / γ) * ((1 - δ) * k_vec[t-2] + A * (k_vec[t-2]^α) - k_vec[t-1])
        k_next = term1 - term2

        kt = k_vec[t-1]
        ktp = k_next
        tvc_vec[t-1] = tvc_compute(kt, ktp, (t-1))

        # Steady-state convergence condition
        if t > 90
            ss_cond = abs(mean(k_vec[t-3:t-1]) - k_next) < 1e-4
            if ss_cond 
                k_next = round(k_next, digits = 8)
            end
        end

        if k_next < 0
            return k_vec[1:(t-1)], tvc_vec[1:(t-1)]
        else
            k_vec[t] = k_next
        end
    end

    return k_vec, tvc_vec
end

# Initial conditions
k0 = k_ss / 4
T = 250

# Wrapper for TVC minimization
function tvc_wrapper(k1)
    k_vec_guess, tvc_vec = k_end(k0, k1, T)
    return abs(tvc_vec[end]) * 1e-5
end


# Wrapper for capital steady-state minimization
function kpr_wrapper(k1_u, k1_d, max_iter)
    iter = 0     
    tol = 1e-5
    diff = 10 

    while diff > tol

        k1_guess = (k1_u + k1_d) / 2 
        k_vec_u, _ = k_end(k0, k1_guess, T)

        if k_vec_u[end] > k_ss # overshoot
            k1_u = (k1_u + k1_d) / 2
        else 
            k1_d = (k1_u + k1_d) / 2
        end
        
        diff = abs(k_vec_u[end] - k_ss)  * 1e-2
        if mod(iter, 2) == 0
            println("Current Diff in $iter: ", diff)
        end

        @assert iter <= max_iter "Error"
        iter = iter + 1
    end
    println("Successful shooting")
    return (k1_u + k1_d) / 2
end

k1_sol = kpr_wrapper(3, 2.8, 500)

# bisection


# undershoot:
k_vec_us, tvc_vec_us = k_end(k0, k1_sol * 0.9, T)

# overshoot:
k_vec_os, tvc_vec_os = k_end(k0, k1_sol * 1.1, T)

k_vec_optim, tvc_vec_optim = k_end(k0, k1_sol, T)

# Plotting capital evolution
plot(1:length(k_vec_us), k_vec_us, 
    xlabel = "Time", ylabel = "Capital Stock", 
    label = "Undershoot", lw = 3, color = :red, alpha = 0.7)

plot!(1:length(k_vec_os), k_vec_os, label = "Overshoot", xlabel = "Time", color = :red, alpha = 0.5, ylabel = "Capital Stock", lw = 3)
plot!(1:length(k_vec_optim), k_vec_optim, label = "Optimal Path", color = :blue, lw = 3)

savefig("k_vec_paths.png")



plot(1:length(tvc_vec_us), log.(tvc_vec_us), 
    xlabel = "Time", ylabel = "log(TVC)", 
    label = "Undershoot", lw = 3, color = :red, alpha = 0.7)

plot!(1:length(tvc_vec_os), log.(tvc_vec_os), label = "Overshoot", xlabel = "Time", color = :red, alpha = 0.5, ylabel = "log(TVC)", lw = 3)
plot!(1:length(tvc_vec_optim), log.(tvc_vec_optim), label = "Optimal Path", color = :blue, lw = 3)

savefig("tvc_overtime.png")


# Compare to the result obtained from VFI
include("VFI_function.jl")

computational_parameters = (I = 800, max_iter = 1200)
model_param0 = (
    γ = 2,
    δ = 0.05,
    β = 0.98,
    α = 1/3,
    A = 1
)

# Compare the Shooting solution to the VFI result
## baseline 
k_vec, v_sol, kpr_sol = VFI(model_param0, computational_parameters)

# Derive optimal path based on kpr_sol_0
k_path = zeros(T+1)
k_path[1] = k_ss / 4
for t in 2:(T+1)
    idx = argmin(abs.(k_vec .- k_path[t-1]))
    k_path[t] = kpr_sol[idx]
end



# Plotting capital evolution
plot(1:length(k_path), k_path, 
    xlabel = "Time", ylabel = "Capital Stock", 
    label = "VFI", lw = 3, color = :red, alpha = 0.7)

plot!(1:length(k_vec_optim), k_vec_optim, label = "Shooting Algorithm", xlabel = "Time", color = :blue, alpha = 0.8, ylabel = "Capital Stock", lw = 3)

plot!(1:length(k_vec_optim), abs.(k_vec_optim .- k_path), label = "Abs Difference", xlabel = "Time", color = :black, alpha = 0.2, ylabel = "Capital Stock", lw = 3)

plot!(legend = :right)


savefig("k_vec_paths_comparison.png")



