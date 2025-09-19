using Optim, Distributions, Plots, Random, Parameters, LinearAlgebra, Roots, ForwardDiff
using Interpolations 


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
        term1 = ((1 - δ) *  k_vec[t-1] + A * k_vec[t-1]^α)
        term2 = (β^(1/γ)) * (1 - δ + A * α * k_vec[t-1]^(α-1))^(1/γ) * ((1-δ) * k_vec[t-2] + A * k_vec[t-2]^ α - k_vec[t-1])
        k_vec[t] = term1 + term2
    end
    return k_vec
end


function TVC_val(k_vec)
    T = length(k_vec) - 1
    term1 = ((1 - δ) * k_vec[end-1] + A * k_vec[end - 1]^α - k_vec[end])^(-γ)
    term2 = 1 - δ + A * α * k_vec[end-1]^(α - 1)
    val = β^T * term1 * term2 * k_vec[end-1]
    return val / 1e10
end

# Use a solver to find optimal k1 given k0 
k0 = 1
T = 500

function wrapper(k1)
    k_vec_guess = k_end(k0, k1, T)
    obj = TVC_val(k_vec_guess)
    return obj 
end

# Use BFGS to find the minimum, set tol = 1e-3

