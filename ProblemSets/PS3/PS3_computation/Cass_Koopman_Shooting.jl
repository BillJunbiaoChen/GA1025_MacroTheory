using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra

γ = 2 
δ = 0.05
β = 0.98
α = 1/3
A = 1 

function k_end(k0, k1, T)
    k_vec = ones(T+1)
    k_vec[1] = k0
    k_vec[2] = k1

    for t in 3:(T+1)
        term1 = (1 - δ) *  k_vec[t-1] + A * (k_vec[t-1]^α)
        term2 = (β^(1/γ)) * (1 - δ + A * α * k_vec[t-2]^(α-1))^(1/γ) * ((1-δ) * k_vec[t-2] + A * (k_vec[t-2]^α) - k_vec[t-1])
        @assert (term1 - term2) >= 0 "Error: Capital should be non-negative"
        k_vec[t] = term1 - term2
    end

    tvc_term1_vec = ((1 - δ) .* k_vec[1:end-1] + A .* (k_vec[1:end - 1] .^ α) - k_vec[2:end]) .^ (-γ)
    tvc_term2_vec = 1 .- δ .+ A .* α .* (k_vec[1:end-1] .^ (α - 1))
    tvc_vec = β .^ (1:T) .* tvc_term1_vec .* tvc_term2_vec .* k_vec[1:end-1]
    return k_vec, tvc_vec
end

# Use a solver to find optimal k1 given k0 
k0 = 0.1
T = 200

function wrapper(k1)
    k_vec_guess, tvc_vec = k_end(k0, k1, T)
    return tvc_vec[end] 
end

# Use L-BFGS to find the minimum, set tol = 1e-3
result = optimize(wrapper, 1, 100, L-BFGS(), 
    Optim.Options(g_tol = 1e-2, show_every = 50)
)

println("Optimal k1: ", result.minimizer)
println("Minimum value: ", result.minimum)