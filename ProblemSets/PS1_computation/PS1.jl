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