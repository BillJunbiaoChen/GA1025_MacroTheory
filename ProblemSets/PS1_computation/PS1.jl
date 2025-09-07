using Distributions, Plots, Random, Parameters, LinearAlgebra


T = 50
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


# Question 3.9 Apply Backward Induction to obtain value functions
v_mat = zeros(J, (T+1))
v_mat[:, T+1] = max.(c.* ones(J), w_vec)


for t in T:-1:1
    println("Solving value function at time-", t)
    EV_tprime = sum(v_mat[:, t+1] .* f_vec)
    accept_val = (1 - β^(T+2-t)) / (1 - β) .* w_vec
    reject_val = c .* ones(J) .+ β .* EV_tprime
    v = max.(accept_val, reject_val)
    v_mat[:, t] = v
end


# Visualize value functions year by year
colors = [color for color in cgrad(:viridis, T+1).colors]  # Sample specific colors from the colormap

plot(w_vec, v_mat[:, 1], color=colors[1], xlabel="Wage", ylabel="Value Function", title="Value Functions Over Time", legend=false)
for t in 2:(T+1)
    plot!(w_vec, v_mat[:, t], color=colors[t], legend=false)
end

savefig("value_functions_over_time.png")


# Find the kink for each year
v_mat_inc = zeros(J, T+1)
v_w̄ = zeros(T+1)

for i in 2:J
    v_mat_inc[i, :] = v_mat[i, :] - v_mat[(i-1), :]
end

for t in 1:(T+1)
    idx = sum(v_mat_inc[:, t] .< 1e-10)
    v_w̄[t] = v_mat[:, t][idx]
end



# Combine Value Functions with Reservation Wages
colors = [color for color in cgrad(:viridis, size(v_mat, 2)).colors]

# Base plot
plot(
    w_vec,
    v_mat[:, 1],
    color=colors[1],
    xlabel="Wage",
    ylabel="Value Function",
    title="Value Functions and Reservation Wage Over Time",
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
