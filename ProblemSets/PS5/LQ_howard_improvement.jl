using LinearAlgebra
cd("/Users/junbiao/Dropbox/PhD_lectures/GA1025_MacroTheory/ProblemSets/PS5")
include("doublej.jl")

# Howard Improvement
function policy_func_iteration(R, F, Q, H, β, A, B, F, n)
    tol = 1e-10
    diff = 10
    iter = 1
    max_iter = 1000

    P_old = zeros(n, n)
    F_old = zeros(n, n)

    while diff > tol 
        Y = R .+ F_old' * Q * F_old .+ F_old' * H .+ H * F_old'
        Ã = A .- B * F_old
        P_imp = doublej(β, Ã, Y)
        temp = Q .+ β .* B' * P_imp * B
        F_imp = β .* inv(temp) * (B' * P_imp * A .+ H)
        diff = maximum(abs.(P_imp .- P_old))

        iter = iter + 1 
        @assert iter < max_iter
        P_old = P_imp
        F_old = F_imp
    end
    return P_old, F_old
end
