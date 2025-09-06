# This script implements an extended McCall job search model with endogenous job offers.


using Random, Distributions, Plots

β = 0.95  # Discount factor
c = 10.0   # Unemployment benefit 

# ----------------------------- 
# Discretized wage distribution
# ----------------------------- 
μ = 10.0  # Mean for the normal distribution
σ = 2.0   # std for the normal distribution

# Truncate the normal distribution to only non-negative values
truncated_normal = Truncated(Normal(μ, σ), 0, Inf)
num_states = 21
values = 1:num_states 

# Compute probabilities for each discrete value
probabilities = [cdf(truncated_normal, x) - cdf(truncated_normal, x-1) for x in values]

# Normalize probabilities to ensure they sum to 1
probabilities ./= sum(probabilities)


# ------------
# VFI 
# ------------

v_old = zeros(num_states)
v_new = zeros(num_states)
v_imp = zeros(num_states)

policy = zeros(Int, num_states) # We use Int because the actions are indexed by Integers

w_vec = values

max_iter = 1_000
damp = 0.1
tol = 1e-8
diff = 10
iter = 0

while diff > tol 

    EV = probabilities' * v_old
    value_fs = hcat(ones(num_states) .* (c .+ β .* EV), w_vec ./ (1 - β)) 
    v_imp = max.((c .+ β .* EV), w_vec ./ (1 - β))
    policy = [argmax(row) for row in eachrow(value_fs)]
    policy = policy .- 1.0 # policy = 0 for rejecting the wage
    v_new = damp * v_imp + (1 - damp) * v_old

    diff = maximum(abs.(v_new - v_old))

    v_old = copy(v_new)

    iter = iter + 1

    if mod(iter, 100) == 0
        println("Iteration: $iter, Max Diff: $diff")
    end

    @assert iter <= max_iter "Error: Reached Maxi Iteration"

end

# Plot Value function and policy function

v_fig = plot(w_vec, v_new, label="Value Function", xlabel="Wage", ylabel="Value", title="McCall Job Search Model")
savefig(v_fig, "value_function.png") 

policy_fig = scatter(w_vec, policy, 
    label="Policy Function", 
    xlabel="Wage", 
    ylabel="Policy", 
    title="McCall Job Search Model",
    yticks = ([0, 1], ["Reject", "Accept"]))
savefig(policy_fig, "policy_function.png")