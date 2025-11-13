using OrdinaryDiffEq
using Plots
using StaticArrays

"""
    main(; kwargs...)

Run a 1D simulation of the (A, Q) system for blood-flow-like dynamics using
an Lax-finite-volume method in space and SciML's `ODEProblem` in time (RK4).

We model a compliant vessel of length `L` cm, discretized into `N` cells.
The simulation runs over the time span `tspan` seconds.
The vessel has a reference cross-sectional area `A₀` cm², stiffness `β` dyn/cm²,
blood density `ρ` g/cm³, and friction coefficient `friction` dyn·s/cm⁵.
The inflow velocity at the inlet is specified by `inflow_velocity` cm/s.
The solution is saved at intervals of `saveat` seconds.

The function returns a named tuple containing:
- `x`: spatial grid points (cell centers)
- `sol`: the solution object from the ODE solver
- `params`: parameters used in the simulation
"""
function main_one(; L=10.0, N=200, tspan=(0.0, 0.5), A₀=1.0, β=2.0e5,
               ρ=1060.0, friction=50.0, inflow_velocity=0.1, saveat=0.001, plot_solution=true)

    Δx = L / N
    x = range(Δx/2, length=N, step=Δx)

    Ain(t) = A₀ * (1 + 0.05 * sin(2π * t))
    Qin(t) = inflow_velocity * Ain(t)
    Aout(t) = A₀
    Qout(t) = inflow_velocity * A₀

    params = AQparams(N, Δx, A₀, β, ρ, friction, Ain, Qin, Aout, Qout)

    u0 = zeros(2N)
    @inbounds for j in 1:N
        u0[2j-1] = A₀ * (1 + 0.05 * exp(-((x[j] - L/2)^2) / (0.1 * L)^2))
        u0[2j] = inflow_velocity * u0[2j-1]
    end

    prob = ODEProblem(aq_rhs!, u0, tspan, params)
    sol = solve(prob, Tsit5(); saveat=saveat)

    if plot_solution
        plt = plot(layout = (2, 1), size = (900, 600))
        for (i, t) in enumerate(sol.t)
            A_sol = @view sol.u[i][1:2:2N-1]
            Q_sol = @view sol.u[i][2:2:2N]

            plot!(plt[1], x, A_sol; label = i == 1 ? "Area A" : "", xlabel = "x", ylabel = "A", title = "Cross-sectional Area over Time (cm²)")
            plot!(plt[2], x, Q_sol; label = i == 1 ? "Flow Q" : "", xlabel = "x", ylabel = "Q", title = "Flow Rate over Time (cm³/s)")
        end
        display(plt)
    end


    return (; x, sol, params)
end

struct AQparams{F1,F2,F3,F4}
    N::Int
    Δx::Float64
    A₀::Float64
    β::Float64
    ρ::Float64
    friction::Float64
    Ain::F1
    Qin::F2
    Aout::F3
    Qout::F4
end

pressure(A, p) = p.β * (sqrt(A) - sqrt(p.A₀))
pressure_derivative(A, p) = p.β / (2 * sqrt(A))

function wave_speed(A, Q, p)
    u = Q / A
    c = sqrt(max(0, A / p.ρ * pressure_derivative(A, p)))
    return abs(u) + c
end

function numerical_flux(Al, Ql, Ar, Qr, p)
    Fl₁, Fl₂ = Ql, Ql^2 / Al + pressure(Al, p)
    Fr₁, Fr₂ = Qr, Qr^2 / Ar + pressure(Ar, p)

    λ = max(wave_speed(Al, Ql, p), wave_speed(Ar, Qr, p))

    Ĥ₁ = 0.5 * (Fl₁ + Fr₁) - 0.5 * λ * (Ar - Al)
    Ĥ₂ = 0.5 * (Fl₂ + Fr₂) - 0.5 * λ * (Qr - Ql)

    return Ĥ₁, Ĥ₂
end

function aq_rhs!(du, u, p::AQparams, t)
    N = p.N
    Δx = p.Δx

    A = @view u[1:2:2N-1]
    Q = @view u[2:2:2N]

    duA = @view du[1:2:2N-1]
    duQ = @view du[2:2:2N]

    Aext = similar(A, N + 2)
    Qext = similar(Q, N + 2)

    Aext[2:N+1] .= A
    Qext[2:N+1] .= Q

    Aext[1] = p.Ain(t)
    Qext[1] = p.Qin(t)
    Aext[end] = p.Aout(t)
    Qext[end] = p.Qout(t)

    fluxA = zeros(eltype(u), N + 1)
    fluxQ = zeros(eltype(u), N + 1)

    @inbounds for i in 1:N+1
        Ĥ₁, Ĥ₂ = numerical_flux(Aext[i], Qext[i], Aext[i+1], Qext[i+1], p)
        fluxA[i] = Ĥ₁
        fluxQ[i] = Ĥ₂
    end

    @inbounds for j in 1:N
        duA[j] = -(fluxA[j+1] - fluxA[j]) / Δx
        duQ[j] = -(fluxQ[j+1] - fluxQ[j]) / Δx - p.friction * Q[j] / A[j]
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
    @info "Simulation complete" result
end
