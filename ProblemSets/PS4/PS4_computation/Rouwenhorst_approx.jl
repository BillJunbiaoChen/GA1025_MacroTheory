

function rouwenhorst(ρ, N; σ_z = 1.0)
    println("σ_z :", σ_z)
    p = (1 + ρ) / 2
    q = (1 + ρ) / 2
    ψ = sqrt(N - 1) * σ_z

    mat_list = [] 
    push!(mat_list, [p (1-p); (1 - q) q])


    for i in 2:(N-1)
        mat_prev = mat_list[i-1]
        n = size(mat_prev)[1]
        part1 = [mat_prev Int.(zeros(n, 1)); Int.(zeros(1, n+1))]
        part2 = [Int.(zeros(n, 1)) mat_prev; Int.(zeros(1, n+1))]
        part3 = [Int.(zeros(1, n+1)); mat_prev Int.(zeros(n, 1))]
        part4 = [Int.(zeros(1, n+1)); Int.(zeros(n, 1)) mat_prev]
        mat_update = p .* part1 .+ (1-p) .* part2 .+ (1-q) .* part3 .+ q .* part4
        mat_update = mat_update ./ sum(mat_update, dims = 2)
        push!(mat_list, mat_update)
    end

    trans_mat = mat_list[end]
    states = -ψ:(2 * ψ/(N-1)): ψ

    # @assert size(trans_mat) == (length(states), length(states))
    
    return trans_mat, states 
end
