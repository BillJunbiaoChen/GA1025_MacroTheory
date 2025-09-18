using Distributions, Plots, Random, Parameters, LinearAlgebra, Roots, ForwardDiff
using Interpolations 

cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS2/PS2_computation/")

β = 0.97
δ = 0.1
A = 1.7 # Productivity of learning
α = 0.9 # Human capital investment elasticity

N = 100
f_vec = fill(1/N, N)

μ_max = 1.978
ϕ_max = 0.99
μ_vec = 0.002: (μ_max - 0.002) / (N-1): μ_max # Human capital value
ϕ_vec = 0.01:0.02:1 # Human capital investment
s_vec = 0.01:0.01:1 # On-the job search effort

choice_mat = ones(length(ϕ_vec)* length(s_vec), 4)
ϕ_long = kron(ϕ_vec, ones(length(s_vec)))
s_long = kron(ones(length(ϕ_vec)), s_vec)
choice_mat[:, 3] .= ϕ_long
choice_mat[:, 4] .= s_long

dropped_idx = (ϕ_long .+ s_long) .> 1.0
choice_mat = choice_mat[.!dropped_idx, :]

v_old = μ_vec
v_new = similar(v_old)
v_imp = similar(v_old)
damp = 0.5

# Value Function Iteration
diff = 10 
tol = 5e-5

iter = 0 
max_iter = 2000
v_imp_mat = zeros(N, length(s_vec))

s_policy = zeros(N)
ϕ_policy = zeros(N)

while diff > tol
    
    itp = linear_interpolation(μ_vec, v_old[:], extrapolation_bc=Flat())
    flow_util_mat = μ_vec .* ((1 .- choice_mat[:,3] .- choice_mat[:,4])')
    μ_next_mat = A .* (μ_vec .* choice_mat[:,3]').^ α .- δ .* μ_vec
    
    π_long = choice_mat[:,4]' .^ (0.5)

    # handle over-float states at the boundary 
    μ_next_mat[μ_next_mat .<= 0] .= 0.0
    μ_next_mat[μ_next_mat .>= μ_max] .= μ_max

    v_next_mat = itp.(μ_next_mat)

    # helpfer function to compute ∑ fₘ * max(v(μₘ), v(x')), given the guess of VF
    function EV(v_next)
        v_vec_max = max.(v_old, v_next)
        val = mean(v_vec_max)
        return val
    end

    EV_vec_next = EV.(μ_next_mat)

    RHS = flow_util_mat .+ β .* (π_long .* EV_vec_next .+ (1 .- π_long) .* v_next_mat)
    ϕ_policy = choice_mat[:,3][getindex.(argmax(RHS, dims = 2), 2)]
    s_policy = choice_mat[:,4][getindex.(argmax(RHS, dims = 2), 2)]
    
    v_imp = maximum(RHS, dims = 2)
    v_new = damp .* v_imp .+ (1 .- damp) .* v_old

    # Compute diff 
    diff = maximum(abs.(v_old ./ v_new .- 1))
    
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
    lw = 3,
    title = "Value Function")
    
savefig("value_function.png")



# Visualize the policy functions
plot(μ_vec, s_policy, xlabel = "Human Capital", ylabel = "Search Effort", 
    legend = false,
    lw = 3,
    title = "Search Policy Function")
    
savefig("serach_policy_function.png")

plot(μ_vec, ϕ_policy, xlabel = "Human Capital", ylabel = "Human Capital Investment", 
    legend = false,
    lw = 3,
    title = "Human Capital Investment Policy")
    
savefig("hk_investment_policy_function.png")


