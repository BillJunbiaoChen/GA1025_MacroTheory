
function VFI_stochastic(model_params, computational_params)
    iter = 0
    @unpack γ, δ, β, α, A, ρ, σ_w, P, z_vec, k_vec = model_params
    @unpack diff, tol, max_iter, damp = computational_params
    σz_sqr = (σ_w^2)/(1 - (ρ^2))
    N = length(z_vec)
    K = length(k_vec)

    v_old = zeros(K*N, 1) # k goes first b/c Julia is column-major
    k_policy = zeros(K*N, 1)

    while diff > tol 
        # Matrix-tization for the Bellman operator
        v_mat_old = reshape(v_old, N, K)
        EV_mat = P * v_mat_old

        k_prod = kron(k_vec .^ α, A .* exp(-0.5 * σz_sqr) .* (exp.(z_vec)))
        k_remaining = kron(((1 - δ) .* k_vec), ones(N))
        c_mat = k_prod .+ k_remaining .- k_vec'

        # identify negative consumption 
        infeasible_idx = c_mat .< 0.0

        # Flow utility 
        u_mat = (c_mat .^ (1 - γ) .- 1) /(1 - γ)
        u_mat[infeasible_idx] .= -Inf

        # implement Bellman 
        EV_mat_long = kron(EV_mat, ones(K))
        v_imp = maximum(u_mat .+ β .* EV_mat_long, dims = 2)
        k_policy = k_vec[getindex.(argmax(u_mat .+ β .* EV_mat_long, dims = 2), 2)]

        if iter < 20
            v_new = damp .* v_imp .+ (1 .- damp) .* v_old
        else
            v_new = v_imp
        end

        diff = maximum(abs.(v_new .- v_old))
        v_old = v_new

        iter = iter + 1

        if mod(iter, 20) == 1
            println("Diff in iteration $iter:", diff)
        end

        @assert iter <= max_iter "Error: Reached Maximum iteration"
    end
    return v_old, k_policy
end

