using LinearAlgebra
cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS5")
include("doublej.jl")

R = [-0.5 0 0 0 30;0 0 0 0 0;0 0 0 0 0;0 0 0 0 0;30 0 0 0 -1800]
A = [0.8 -0.3 0.2 0 1; 1 0 0 0 0; 0 0 0.7 -0.2 0; 0 0 1 0 0; 0 0 0 0 1]

# Exercise 2.15 (i)
C = [1 0; 0 0; 0 1; 0 0; 0 0]
β = 0.95
P = doublej(β, A, R)
σ = (β/(1-β)) * tr(C' * P * C);

# println("P = ", round.(P, digits = 3))
# println("σ = ", round.(σ, digits = 3))


# Exercise 2.15 (ii)
C = [2 0; 0 0; 0 1; 0 0; 0 0]
β = 0.95
P = doublej(β, A, R)
σ = (β/(1-β)) * tr(C' * P * C);

println("P = ", round.(P, digits = 3))
println("σ = ", round.(σ, digits = 3))
