cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS3/PS3_computation")

using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

γ = 2 
δ = 0.05
β = 0.98
α = 1/3
A = 1 

# compute the capital level in ss
k̄ = ((1 - β) + (β * δ)/(β * A * α))^(1/(α - 1))

# Helper functions 

# Function to plot k_vec (entire time horizon)
function plot_k_vec(k_vec_list, colors, alphas)
    for (i, k_vec) in enumerate(k_vec_list)
        plot!(1:length(k_vec), k_vec,
            xlabel = "Time", ylabel = "Capital Stock",
            legend = false, lw = 2, color = colors[i], alpha = alphas[i])
    end
end

# Function to plot tvc_vec (last 50 periods with adjusted x-ticks)
function plot_tvc_vec(tvc_vec_list, colors, alphas, total_time)
    for (i, tvc_vec) in enumerate(tvc_vec_list)
        # Generate x-ticks for the last 50 periods
        x_ticks = (total_time - 49):total_time
        plot!(x_ticks, tvc_vec[end-49:end],  # Only last 50 periods
            xlabel = "Time", ylabel = "TVC",
            legend = false, lw = 2, color = colors[i], alpha = alphas[i])
    end
end

# Computation Functions
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
T = 250

function wrapper(k1)
    k_vec_guess, tvc_vec = k_end(k0, k1, T)
    return tvc_vec[end] 
end

# undershoot:
k_vec_us_1, tvc_vec_us_1 = k_end(k0, 1.292, T)
k_vec_us_2, tvc_vec_us_2 = k_end(k0, 1.2932, T)
k_vec_us_3, tvc_vec_us_3 = k_end(k0, 1.2934, T)
k_vec_us_4, tvc_vec_us_4 = k_end(k0, 1.2936, T)



# overshoot:
k_vec_os_1, tvc_vec_os_1 = k_end(k0, 1.295, T)
k_vec_os_2, tvc_vec_os_2 = k_end(k0, 1.296, T)
k_vec_os_3, tvc_vec_os_3 = k_end(k0, 1.298, T)

# overshoot:
k_vec_optim, tvc_vec_optim = k_end(k0, 1.2938943267, T)

# Plotting k_vec
plot(1:length(k_vec_us_1), k_vec_us_1, 
    xlabel = "Time", ylabel = "Capital Stock", 
    legend = false, lw = 2, color = :blue, alpha = 0.9)

plot_k_vec([k_vec_us_2, k_vec_us_3, k_vec_us_4], [:blue, :blue, :blue], [0.8, 0.6, 0.5])
plot_k_vec([k_vec_os_1, k_vec_os_2, k_vec_os_3], [:red, :orange, :orange], [0.5, 1.0, 0.8])
plot_k_vec([k_vec_optim], [:green], [1.0])

savefig("k_overtime.png")

# Plotting tvc_vec with adjusted x-ticks
plot((T - 49):T, tvc_vec_us_1[end-49:end],  # Adjust x-ticks
    xlabel = "The Last 50 Periods", ylabel = "Transversality conditions", 
    legend = false, lw = 2, color = :blue, alpha = 0.9)

plot_tvc_vec([tvc_vec_us_2, tvc_vec_us_3, tvc_vec_us_4], [:blue, :blue, :blue], [0.8, 0.6, 0.5], T)
plot_tvc_vec([tvc_vec_os_1, tvc_vec_os_2, tvc_vec_os_3], [:red, :orange, :orange], [0.5, 1.0, 0.8], T)
plot_tvc_vec([tvc_vec_optim], [:green], [1.0], T)

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
