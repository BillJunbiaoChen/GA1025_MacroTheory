using Distributions, Plots, Random, Parameters, LinearAlgebra


T_vec = [50, 100, 200, 500, 1000]
β = 0.97
B = 1
c = 0.3


w_vec = (0:(1/100):1)

J = length(w_vec)

# Discrete a Beta Function (2, 2)
beta_dist = Beta(2, 2)

# Construct probabilities 
f_vec = zeros(J)
f_vec[1] = cdf(beta_dist, 0.5* (w_vec[1] + w_vec[2]))

for j in 2:(J-1)
    p1 = 0.5* (w_vec[j+1] + w_vec[j])
    p2 = 0.5* (w_vec[j] + w_vec[j-1])
    f_vec[j] = cdf(beta_dist, p1) - cdf(beta_dist, p2)
end

f_vec[J] = 1 - cdf(beta_dist, 0.5* (w_vec[J] + w_vec[J-1]))
Ew = sum(w_vec .* f_vec)

# Question 3.8 Compute reservation wages 
w̄_vec = zeros((T+1))
w̄_vec[T+1] = c # w̄_T = c 


for t in T:-1:1
    reject_idx = w_vec .< w̄_vec[t+1]
    p_rej = sum(f_vec[reject_idx])
    term1 = w̄_vec[t+1] * p_rej
    term2 = sum(f_vec[reject_idx] .* w_vec[reject_idx])
    w̄_vec[t] = (c * (1 - β) / (1 - β^(T-t+2))) +
               (β * (1 - β^(T-t+1)) / (1 - β^(T-t+2)) * (Ew + term1 - term2))
end


# Visualize reservation wages
plot((1:(T+1)), w̄_vec, xlabel="Time", ylabel="Reservation wage", legend=false)
savefig("reservation_wages_over_time.png")


# Combine Value Functions with Reservation Wages
colors = [color for color in cgrad(:viridis, size(v_mat, 2)).colors]

# Base plot
plot(
    w_vec,
    v_mat[:, 1],
    color=colors[1],
    xlabel="Wage",
    # ylabel="Value Function",
    title="The Convergence of Reservation Wage",
    legend=false,
    titlefont=10, guidefont=8, tickfont=6
)


for t in 2:size(v_mat, 2)
    plot!(w_vec, v_mat[:, t], color=colors[t], label=false)
end


plot!(w̄_vec, v_w̄, color=:blue, lw=2)
x_anno = w̄_vec[end]
y_anno = v_w̄[end]
annotate!(x_anno, y_anno, text("Reservation Wage at t = T", 8, :blue, :left))

x_anno = w̄_vec[1]
y_anno = v_w̄[1]
annotate!(x_anno, y_anno, text("Reservation Wage at t = 1", 8, :blue, :left))

savefig("value_functions_over_time_w_reservation_wage.png")
