function main()
    # Parameters
    J, N = 100, 100
    tⁿ, T = 0.0, 1.0
    Δx = 1 / J
    Δt = 1 / N
    α = J/N # \Dlta t / \Delta x

    # Initialize grid
    x_mesh = range(0, 1, length=N+1)

    # Initial condition
    u⁰(xⱼ) = xⱼ*(1 - xⱼ)

    uⁿ = [u⁰(x) for x in x_mesh]

    while tⁿ < T
        tⁿ += Δt

        # Use boundary conditions
        uⁿ[1], uⁿ[end] = 0.0, 0.0

        # Update uⁿ using upwind scheme
        for j in 2:J
            uⁿ[j] = uⁿ[j] - (Δt / Δx) * (uⁿ[j] - uⁿ[j-1])
        end

    end

    return uⁿ
end
