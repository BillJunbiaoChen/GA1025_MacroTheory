

using Random, Distributions, Plots
cd("/Users/junbiao/Dropbox/PhD_firstyear/GA1025_MacroTheory/Computation")

β = 0.95  # Discount factor
A_vec = [0.1, 1, 2, 3, 10, 15]
N = 500 # number of grid points
v_mat = zeros(N, length(A_vec))

tol = 1e-8
max_iter = 1_000
h_vec = 0:0.5:(N-1)*0.5
diff = 10.0
iter = 0
v_old = zeros(N) 

for a in 1:length(A_vec)
    A = A_vec[a]

    while diff > tol && iter < max_iter
        v_next = vcat(v_old[3:end], v_old[end], v_old[end])
        v_imp = max(A .* h_vec, β .* v_next)

        diff = maximum(abs.(v_imp - v_old))
        if mod(iter, 100) == 0
            println("A: $A, Iteration: $iter, Max Diff: $diff")
        end
        v_old = v_imp
        iter += 1
    end
    v_mat[:, a] .= v_old[:]
end


# plot value functions for all A values
plot(h_vec, v_mat[:, 1], label = "A = $(A_vec[1])", xlabel = "h", ylabel = "Value Function", title = "Value Functions for Different A Values")
for a in 2:length(A_vec)
    plot!(h_vec, v_mat[:, a], label = "A = $(A_vec[a])")
end
display(plot!)

savefig("treecutting_value_functions.png")