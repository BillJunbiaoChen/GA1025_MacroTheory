using Random, Distributions, Plots

cd("/Users/junbiao/Dropbox/PhD_firstyear/GA1025_MacroTheory/Computation")

β = 0.95  # Discount factor
A_vec = [2]
N = 1000 # number of grid points
h_vec = collect(0:0.5:(N - 1) * 0.5)
v_mat = zeros(N, length(A_vec))
policy_mat = zeros(Int, N, length(A_vec))

tol = 1e-8
max_iter = 1_000

g_unit = 1

for (a_idx, A) in enumerate(A_vec)
    v_old = zeros(N)
    policy_old = zeros(Int, N)
    diff  = Inf
    iter  = 0
    damp = 0.1
    while diff > tol && iter < max_iter
        v_next = vcat(v_old[(g_unit + 1):end], kron(ones(g_unit), v_old[end]))
        v_imp = max(A .* h_vec, β .* v_next)
        policy_old = 2 .- (A .* h_vec .>= β .* v_next)  .|> Int

        diff = maximum(abs.(v_imp .- v_old))
        if mod(iter, 100) == 0
            println("A: $A, Iteration: $iter, Max Diff: $diff")
        end

        v_old = damp * v_imp + (1 - damp) * v_old
        iter += 1
    end

    v_mat[:, a_idx] .= v_old
    policy_mat[:, a_idx] .= policy_old
end

plt = plot(h_vec, v_mat[:, 1],
           label = "A = $(A_vec[1])",
           xlabel = "h",
           ylabel = "Value Function",
           title = "Value Functions for Different A Values")
for a_idx in 2:length(A_vec)
    plot!(plt, h_vec, v_mat[:, a_idx], label = "A = $(A_vec[a_idx])")
end

display(plt)
savefig("treecutting_value_functions.png")


plt2 = plot(h_vec, policy_mat[:, 1],
           label = "A = $(A_vec[1])",
           xlabel = "h",
           ylabel = "Policy Function",
           title = "Policy Functions for Different A Values")
for a_idx in 2:length(A_vec)
    plot!(plt2, h_vec, policy_mat[:, a_idx], label = "A = $(A_vec[a_idx])")
end

display(plt2)
savefig("treecutting_policy_functions.png")