using Distributions, Plots, Random, Parameters, LinearAlgebra, Roots, ForwardDiff
using Interpolations

β = 0.97
B = 1
δ = 0.05
A = 1 # Productivity of learning
α = 0.9 # Human capital investment elasticity

N = 495
f_vec = fill(1/N, N)

μ_vec = 0.002: 0.004: 1.978 # Human capital value
ϕ_vec = 0.002: 0.002: 0.99 # Human capital investment
s_vec = [0.1, 0.5] # On-the job search effort

v_old = μ_vec



# Value Function Iteration

while diff > tol

    for (s, s_val) in enumerate(s_vec)
        π = s_val .^ (0.5)

        flow_util = μ_vec .* (1 .- ϕ_vec' .- s_val)
        μ_next_mat = A .* (μ_vec .* ϕ_vec').^ α .- δ .* μ_vec

        itp = linear_interpolation(μ_vec, v_old, extrapolation_bc=Flat())
        V_x_tprime = itp.(μ_next_mat)

        EV = mean(max.(v_old, V_x_tprime), dims = 1)


    end

    iter = iter + 1
    @assert iter < max_iter "Error: Reached Max Iteration"
end



