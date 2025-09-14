using Distributions, Plots, Random, Parameters, LinearAlgebra, Roots


T_vec = [50, 51, 52, 53, 54, 55, 60, 65, 75, 100, 200, 300]
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

# use root-finding algorithm to obtain w̄ in infinite horizon.
g(x) = x - c - (β / (1 - β)) * sum(f_vec[w_vec .> x] .* (w_vec[w_vec .> x] .- x))
w_bar_infinite = find_zero(g, (0, 1), A42())



# Examine w̄ in finite periods with increasing T
w̄T_vec = zeros(length(T_vec))

tt = 1

for T in T_vec
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
    w̄T_vec[tt] = w̄_vec[1]
    tt = tt + 1
end

plot(
    T_vec,
    w̄T_vec,
    color=:blue,
    ylabel="Reservation Wage",
    xlabel="Number of periods",
    label="Finite Horizon"
)


plot!(
    T_vec, 
    ones(length(T_vec)) .* w_bar_infinite, 
    color=:black, 
    lw=2,
    label="Infinite Horizon"
)


savefig("convergence_of_reservation_wage.png")
