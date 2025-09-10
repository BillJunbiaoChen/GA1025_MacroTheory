using Distributions, Statistics
using Plots


c = 0.3
β = 0.6
ϕ1 = 0.1
ϕ2 = 0.5
Ew = 0.5

# x-axis 
x_vec = 0:0.01:1
g1_vec = ((1 - β)/β) * x_vec .-  ((1 - β*ϕ1) * c/β) 
g2_vec = ((1 - β)/β) * x_vec .-  ((1 - β*ϕ2) * c/β)
h_vec = -0.5 * x_vec .^ (0.5) .+ 0.5

# Visualize 
# Base plot
plot(
    x_vec,
    g1_vec,
    color=:blue,
    ylabel="CDF",
    legend=false,
    titlefont=10, guidefont=8, tickfont=6
)


plot!(
    x_vec,
    g2_vec,
    color=:red,
    legend=false
)


plot!(
    x_vec,
    h_vec,
    color=:black,
    legend=false
)

plot!(
    x_vec,
    zeros(length(x_vec)),
    color=:black,
    title="ϕ1 < ϕ2",
    legend=false
)

annotate!((c * (1 - β * ϕ1) / β) + 0.2, 0.02, text("w-bar (ϕ1)", 8, :blue, :left))
annotate!((c * (1 - β * ϕ2) / β) + 0.1, 0.02, text("w-bar (ϕ2)", 8, :red, :left))

savefig("McCall_w_wage_growth.png")
