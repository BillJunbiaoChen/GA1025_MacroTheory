function simulate_MC(y0, states, P, T)
    
    state_idx = zeros(Int, T)
    state_idx[1] = argmin(abs.(states .- y0))

    for t in 2:T
        # Random.seed!(t)
        v = rand(Uniform(0,1))
        cdf = cumsum(@view P[state_idx[t-1], :])
        idx = searchsortedfirst(cdf, v)
        state_idx[t] = idx
    end
    return states[state_idx]
end