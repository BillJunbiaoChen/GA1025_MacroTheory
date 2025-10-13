function doublej(β, A, Y)
    # This function computes the infinite sum V given by:
    #
    #         V = Σ  βʲ * (A'ʲ) * Y * (Aʲ)
    #
    # where a1 and b1 are (n x n) matrices with eigenvalues whose moduli
    # are bounded by unity, and b1 is an (n x n) matrix.
    # The sum goes from j = 0 to j = infinity.
    #

    # In the form of discrete Lyapunov equation:
    # v = Y + βA'vA
    
    # The algorithm iteratively computes:
    #   a(j) = a(j-1) * a(j-1)
    #   v(j) = v(j-1) + a(j-1)' * v(j-1) * a(j-1)
    #
    # where a(0) = β^{1/2} A 
    # v(0) = Y
    # until convergence.

    a0 = β^(0.5) .* A
    v0 = Y
    v1 = similar(v0)
    diff = 5.0  # Initial large difference
    max_iter = 1000
    iter = 1

    while diff > 1e-10
        a1 = a0 * a0
        v1 = v0 + a0' * v0 * a0
        diff = maximum(abs.(v1 .- v0))

        v0 = v1
        a0 = a1

        iter = iter + 1
        if iter > max_iter
            error("Error: Iteration limit exceeded.")
        end
    end

    return v1
end

# # Testing
# A = [0.5 0.1; 0.2 0.4]
# Y = [1.0 0.0; 0.0 1.0]
# β = 0.9 

# # Compute V
# V = doublej(β, A, Y)
# println("Solution V:")
# println(V)