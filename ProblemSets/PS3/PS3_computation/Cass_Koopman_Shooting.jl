using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

γ = 2 
δ = 0.05
β = 0.98
α = 1/3
A = 1 

function tvc_compute(kt, ktp, t)
    tvc_term1 = ((1 - δ) * kt + A * (kt ^ α) - ktp) ^ (-γ)
    tvc_term2 = 1 - δ + A * α * (kt ^ (α - 1))
    tvc = (β^t) * tvc_term1 * tvc_term2 * kt
    return tvc 
end

function k_end(k0, k1, T)
    k_vec = zeros(T+1)
    k_vec[1] = k0
    k_vec[2] = k1

    tvc_vec = zeros(T)
    tvc_vec[1] = tvc_compute(k0, k1, 1) # t = 1
    
    for t in 3:(T+1)
        term1 = (1 - δ) *  k_vec[t-1] + A * (k_vec[t-1]^α)
        term2 = (β^(1/γ)) * (1 - δ + A * α * k_vec[t-1]^(α-1))^(1/γ) * ((1-δ) * k_vec[t-2] + A * (k_vec[t-2]^α) - k_vec[t-1])
        k_next = term1 - term2

        kt = k_vec[t-1]
        ktp = k_vec[t]
        tvc_vec[t-1] = tvc_compute(kt, ktp, (t-1))

        # Capture k_next around the steady state
        if t > 90
            ss_cond = abs(mean(k_vec[t-3:t]) - k_next) < 1e-4
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

# Use a solver to find optimal k1 given k0 
k0 = 1
T = 300

function wrapper(k1)
    k_vec_guess, tvc_vec = k_end(k0, k1, T)
    return tvc_vec[end] 
end

# undershoot:
k_vec_us_1, tvc_vec_us_1 = k_end(k0, 1.292, T)
k_vec_us_2, tvc_vec_us_2 = k_end(k0, 1.2921, T)
k_vec_us_3, tvc_vec_us_3 = k_end(k0, 1.2922, T)
k_vec_us_4, tvc_vec_us_4 = k_end(k0, 1.29221, T)

# overshoot:
k_vec_os_1, tvc_vec_os_1= k_end(k0, 1.295, T)
k_vec_us_2, tvc_vec_us_2 = k_end(k0, 1.296, T)
k_vec_os_3, tvc_vec_os_3  = k_end(k0, 1.298, T)

plot(1:length(k_vec_us_1), k_vec_us_1, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :blue,
    alpha = 0.9)

plot!(1:length(k_vec_us_2), k_vec_us_2, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :blue,
    alpha = 0.8) 

plot!(1:length(k_vec_us_3), k_vec_us_3, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :blue,
    alpha = 0.6) 

plot!(1:length(k_vec_us_4), k_vec_us_4, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :blue,
    alpha = 0.5) 

plot!(1:length(k_vec_os_1), k_vec_os_1, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :red,
    alpha = 0.5)

plot!(1:length(k_vec_os_2), k_vec_os_2, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :orange,
    alpha = 1.0)

plot!(1:length(k_vec_os_3), k_vec_os_3, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false,
    lw = 2,
    color = :orange,
    alpha = 0.8)







savefig("k_overtime.png")





plot(1:length(tvc_vec_us_1), tvc_vec_us_1, 
    xlabel = "Time", ylabel = "Transversality Condition", 
    legend = false,
    lw = 3,
    color = :blue,
    alpha = 1.0)

plot!(1:length(tvc_vec_us_2), tvc_vec_us_2, 
    xlabel = "Time", ylabel = "Transversality Condition", 
    legend = false,
    lw = 3,
    color = :blue,
    alpha = 0.6)

plot!(1:length(tvc_vec_os_1), tvc_vec_os_1, 
    xlabel = "Time", ylabel = "Transversality Condition", 
    legend = false,
    lw = 3,
    color = :orange,
    alpha = 1.0)

plot!(1:length(tvc_vec_os_2), tvc_vec_os_2, 
    xlabel = "Time", ylabel = "Transversality Condition", 
    legend = false,
    lw = 3,
    color = :orange,
    alpha = 0.6)

savefig("tvc_overtime.png")








# # Use BFGS to find the minimum, set tol = 1e-3
# result = optimize(wrapper, 1.294, 1.3, BFGS(), 
#     Optim.Options(g_tol = 1e-2, show_every = 50)
# )

# println("Optimal k1: ", result.minimizer)
# println("Minimum value: ", result.minimum)

# Fixed point iteration to obtain the ss-k 
# function k_update(k)
#     term1 = (1 - δ) *  k + A * (k^α)
#     term2 = (β^(1/γ)) * (1 - δ + A * α * k^(α-1))^(1/γ) * ((1-δ) * k + A * (k^α) -k)
#     k_imp = term1 - term2
#     return k_imp
# end

# k_old = 2

# for i in 1:100 

#     k_imp = k_update(k_old)
#     k_new = 0.1 * k_imp + 0.9 * k_old
#     k_old = copy(k_new)
# end
