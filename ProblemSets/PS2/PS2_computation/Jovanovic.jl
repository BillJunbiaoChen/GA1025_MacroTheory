using Distributions, Plots, Random, Parameters, LinearAlgebra, Roots, ForwardDiff
using Interpolations # Interpolate VF, and I will check the convergence using the sup-norm

cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS2/PS2_computation/")

β = 0.97
δ = 0.1
A = 1.2 # Productivity of learning
α = 0.9 # Human capital investment elasticity

N = 695
f_vec = fill(1/N, N)

μ_vec = 0.002: (1.978 - 0.002) / (N-1): 1.978 # Human capital value
ϕ_vec = 0.002: (0.99 - 0.002) / (N-1): 0.99 # Human capital investment
s_vec = [0.1, 0.5, 0.6] # On-the job search effort

v_old = μ_vec
v_new = similar(v_old)
v_imp = similar(v_old)
damp = 0.5

ϕ_policy_low_s = similar(v_old)
ϕ_policy_high_s = similar(v_old)

# Value Function Iteration
diff = 10 
tol = 1e-6

while diff > tol
    v_imp_mat = zeros(N, length(s_vec))
    itp = linear_interpolation(μ_vec, v_old[:], extrapolation_bc=Flat())

    for (s, s_val) in enumerate(s_vec)
        π = s_val .^ (0.5)

        flow_util = μ_vec .* (1 .- ϕ_vec' .- s_val)
        μ_next_mat = A .* (μ_vec .* ϕ_vec').^ α .- δ .* μ_vec
        V_next_mat = itp.(μ_next_mat)

        EV = mean(max.(v_old, V_next_mat), dims = 1) # because f_n = 1/N ∀ n


        @assert size(EV) == (1, N)

        # Solve for the optim human capital investment 
        obj = flow_util .+ β .* ((1 - π) .* V_next_mat .+ π .* EV) # broad-casting

        # Update value function 
        v_imp_mat[:, s] = maximum(obj, dims = 2)

        # Obtain policy functions
        ϕ_idx = [argmax(row) for row in eachrow(obj)]
        if s_val == minimum(s_vec)
            ϕ_policy_low_s = ϕ_vec[ϕ_idx]
        end 

        if s_val == maximum(s_vec)
            ϕ_policy_high_s = ϕ_vec[ϕ_idx]
        end
    end

    v_imp = maximum(v_imp_mat, dims = 2)
    v_new = damp .* v_imp .+ (1 .- damp) .* v_old

    # Compute diff 
    diff = maximum(abs.(v_old .- v_new))
    
    if mod(iter, 10) == 1
        println("Diff in ", iter, ": ", diff)
    end

    v_old = v_new

    iter = iter + 1
    @assert iter < max_iter "Error: Reached Max Iteration"
end

# Visualize the value function
plot(μ_vec, v_new, xlabel = "Human Capital", ylabel = "Value Function", 
    legend = false,
    title = "Value Function")
    
savefig("value_function.png")

# Visualize the policy functions
plot(μ_vec, ϕ_policy_low_s, label = "Low On-the-job Search Effort", xlabel = "Human Capital", ylabel = "Optimal Human Capital Investment", title = "Policy Function")
plot!(μ_vec, ϕ_policy_high_s, label = "High On-the-job Search Effort")
savefig("policy_function.png")
