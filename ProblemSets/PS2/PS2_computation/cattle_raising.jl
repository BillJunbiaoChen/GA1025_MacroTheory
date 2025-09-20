using Plots

# Define y_t
function y_t(t, a, β, h, T, α)
    R = (a * β)^(1 / (1 - α))
    num = R^(t-1) * (1 - R / a)
    den = 1 - (R^T) / (a^T)
    return h * num / den
end

# Parameters
a   = 1.1
β   = 0.95
h   = 1.0
T   = 10000
α_vals = [0.2, 0.5, 0.8]
styles = [:solid, :dash, :dot]   # line styles for each α

# Time indices
ts = 1:T

# Plot different α values
plt = plot(xlabel="t", ylabel="Cattle sold", title="a * β > 1")
for (i, α) in enumerate(α_vals)
    ys = [y_t(t, a, β, h, T, α) for t in ts]
    plot!(plt, ts, ys, lw=3, linestyle=styles[i], label="α = $α")
end

# Save figure
savefig(plt, "y_t_paths_greater_than_one.png")



# Parameters
a   = 1.01
β   = 0.95
h   = 1.0
T   = 10000
α_vals = [0.2, 0.5, 0.8]
styles = [:solid, :dash, :dot]   # line styles for each α

# Time indices
ts = 1:T

# Plot different α values
plt = plot(xlabel="t", ylabel="Cattle sold", title="a * β < 1")
for (i, α) in enumerate(α_vals)
    ys = [y_t(t, a, β, h, T, α) for t in ts]
    plot!(plt, ts, ys, lw=3, linestyle=styles[i], label="α = $α")
end

# Save figure
savefig(plt, "y_t_paths_smaller_than_one.png")
