function VFI(model_params, comp_params)
    @unpack γ, δ, β, α, A = model_params
    @unpack I, max_iter = comp_params

    k_ss = ((1 - β + β * δ)/(β * A * α))^(1/(α - 1))

    k_min = 0.1 * round(k_ss, digits = 8)
    k_max = 2 * round(k_ss, digits = 8)
    k_vec = k_min:(k_max - k_min)/(I-1) :k_max


    v_old = zeros(I)
    k_policy = zeros(I)
    c_policy = zeros(I)

    iter = 0 
    max_iter = 1000
    tol = 1e-8
    diff = 10 

    while diff > tol && iter < max_iter 

        consump_mat = (1 - δ) .* k_vec + A * (k_vec .^ α) .-  k_vec'

        # select infeasible consumption
        c_max = (1 - δ) * k_max + A * (k_max ^ α)
        infeasible_idx = (consump_mat .< 0) .| (consump_mat .> c_max)

        # compute utility mat 
        util_mat = (consump_mat.^(1 -  γ) .- 1) ./ (1 - γ)
        util_mat[infeasible_idx] .= -Inf

        # update VF 
        v_imp = maximum(util_mat .+ β .* v_old', dims = 2)
        k_policy = k_vec[getindex.(argmax(util_mat .+ β .* v_old', dims = 2), 2)]
        c_policy = (1 - δ) .* k_vec + A * (k_vec .^ α) .- k_policy

        diff = maximum(abs.(v_imp .- v_old))
        v_old = copy(v_imp)

        iter = iter + 1
        if mod(iter, 50) == 0
            println("Diff in $iter is :", diff)
        end
    end

    return k_vec, v_old, k_policy, c_policy
end