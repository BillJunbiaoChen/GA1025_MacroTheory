# This script implements the McCall job search model in Julia.


using Random, Distributions, Plots


# Is VF concave even though the flow utility is linear?
β = 0.97

v_old = zeros(1000)
v_new = similar(v_old)

u_vec = collect(0:0.1:99.9)
max_iter = 10000
diff = 10 
tol = 1e-6 
iter = 0
while diff > tol
    v_new = max.(u_vec .+ β .* v_old)
    v_old = copy(v_new)

    iter = iter + 1
    diff = maximum(abs.(v_new .- v_old))
    @assert iter < max_iter
end

plot(u_vec, v_new, label="Value Function", xlabel="wage offer", ylabel="Value", legend=:topleft)