using OrdinaryDiffEq
using Plots
using StaticArrays

"""
    main_one(; kwargs...)

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

struct AQ2DParams{F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,TX,TY}
    Nx::Int
    Ny::Int
    Δx::Float64
    Δy::Float64
    A₀::Float64
    β::Float64
    ρ::Float64
    friction::Float64
    x::TX
    y::TY
    A_left::F1
    Qx_left::F2
    Qy_left::F3
    A_right::F4
    Qx_right::F5
    Qy_right::F6
    A_bottom::F7
    Qx_bottom::F8
    Qy_bottom::F9
    A_top::F10
    Qx_top::F11
    Qy_top::F12
end

wave_speed_component(A, Qn, p) = abs(Qn / A) + sqrt(max(0, A / p.ρ * pressure_derivative(A, p)))

function numerical_flux_x(Al, Qxl, Qyl, Ar, Qxr, Qyr, p)
    Fl₁ = Qxl
    Fl₂ = Qxl^2 / Al + pressure(Al, p)
    Fl₃ = Qxl * Qyl / Al

    Fr₁ = Qxr
    Fr₂ = Qxr^2 / Ar + pressure(Ar, p)
    Fr₃ = Qxr * Qyr / Ar

    λ = max(wave_speed_component(Al, Qxl, p), wave_speed_component(Ar, Qxr, p))

    Ĥ₁ = 0.5 * (Fl₁ + Fr₁) - 0.5 * λ * (Ar - Al)
    Ĥ₂ = 0.5 * (Fl₂ + Fr₂) - 0.5 * λ * (Qxr - Qxl)
    Ĥ₃ = 0.5 * (Fl₃ + Fr₃) - 0.5 * λ * (Qyr - Qyl)

    return Ĥ₁, Ĥ₂, Ĥ₃
end

function numerical_flux_y(Al, Qxl, Qyl, Ar, Qxr, Qyr, p)
    Gl₁ = Qyl
    Gl₂ = Qxl * Qyl / Al
    Gl₃ = Qyl^2 / Al + pressure(Al, p)

    Gr₁ = Qyr
    Gr₂ = Qxr * Qyr / Ar
    Gr₃ = Qyr^2 / Ar + pressure(Ar, p)

    λ = max(wave_speed_component(Al, Qyl, p), wave_speed_component(Ar, Qyr, p))

    Ĥ₁ = 0.5 * (Gl₁ + Gr₁) - 0.5 * λ * (Ar - Al)
    Ĥ₂ = 0.5 * (Gl₂ + Gr₂) - 0.5 * λ * (Qxr - Qxl)
    Ĥ₃ = 0.5 * (Gl₃ + Gr₃) - 0.5 * λ * (Qyr - Qyl)

    return Ĥ₁, Ĥ₂, Ĥ₃
end

function aq_rhs_2d!(du, u, p::AQ2DParams, t)
    Nx, Ny = p.Nx, p.Ny
    Δx, Δy = p.Δx, p.Δy

    A = reshape(@view(u[1:3:end]), Nx, Ny)
    Qx = reshape(@view(u[2:3:end]), Nx, Ny)
    Qy = reshape(@view(u[3:3:end]), Nx, Ny)

    duA = reshape(@view(du[1:3:end]), Nx, Ny)
    duQx = reshape(@view(du[2:3:end]), Nx, Ny)
    duQy = reshape(@view(du[3:3:end]), Nx, Ny)

    Aext = zeros(eltype(u), Nx + 2, Ny + 2)
    Qxext = similar(Aext)
    Qyext = similar(Aext)

    @inbounds Aext[2:Nx+1, 2:Ny+1] .= A
    @inbounds Qxext[2:Nx+1, 2:Ny+1] .= Qx
    @inbounds Qyext[2:Nx+1, 2:Ny+1] .= Qy

    @inbounds for j in 1:Ny
        yj = p.y[j]
        Aext[1, j+1] = p.A_left(t, yj)
        Qxext[1, j+1] = p.Qx_left(t, yj)
        Qyext[1, j+1] = p.Qy_left(t, yj)

        Aext[end, j+1] = p.A_right(t, yj)
        Qxext[end, j+1] = p.Qx_right(t, yj)
        Qyext[end, j+1] = p.Qy_right(t, yj)
    end

    @inbounds for i in 1:Nx
        xi = p.x[i]
        Aext[i+1, 1] = p.A_bottom(t, xi)
        Qxext[i+1, 1] = p.Qx_bottom(t, xi)
        Qyext[i+1, 1] = p.Qy_bottom(t, xi)

        Aext[i+1, end] = p.A_top(t, xi)
        Qxext[i+1, end] = p.Qx_top(t, xi)
        Qyext[i+1, end] = p.Qy_top(t, xi)
    end

    # Corners
    Aext[1, 1] = p.A_left(t, p.y[1])
    Qxext[1, 1] = p.Qx_left(t, p.y[1])
    Qyext[1, 1] = p.Qy_left(t, p.y[1])

    Aext[1, end] = p.A_left(t, p.y[end])
    Qxext[1, end] = p.Qx_left(t, p.y[end])
    Qyext[1, end] = p.Qy_left(t, p.y[end])

    Aext[end, 1] = p.A_right(t, p.y[1])
    Qxext[end, 1] = p.Qx_right(t, p.y[1])
    Qyext[end, 1] = p.Qy_right(t, p.y[1])

    Aext[end, end] = p.A_right(t, p.y[end])
    Qxext[end, end] = p.Qx_right(t, p.y[end])
    Qyext[end, end] = p.Qy_right(t, p.y[end])

    flux_x_A = zeros(eltype(u), Nx + 1, Ny)
    flux_x_Qx = similar(flux_x_A)
    flux_x_Qy = similar(flux_x_A)

    flux_y_A = zeros(eltype(u), Nx, Ny + 1)
    flux_y_Qx = similar(flux_y_A)
    flux_y_Qy = similar(flux_y_A)

    @inbounds for i in 1:Nx+1
        for j in 1:Ny
            Ĥ₁, Ĥ₂, Ĥ₃ = numerical_flux_x(Aext[i, j+1], Qxext[i, j+1], Qyext[i, j+1],
                                           Aext[i+1, j+1], Qxext[i+1, j+1], Qyext[i+1, j+1], p)
            flux_x_A[i, j] = Ĥ₁
            flux_x_Qx[i, j] = Ĥ₂
            flux_x_Qy[i, j] = Ĥ₃
        end
    end

    @inbounds for i in 1:Nx
        for j in 1:Ny+1
            Ĥ₁, Ĥ₂, Ĥ₃ = numerical_flux_y(Aext[i+1, j], Qxext[i+1, j], Qyext[i+1, j],
                                           Aext[i+1, j+1], Qxext[i+1, j+1], Qyext[i+1, j+1], p)
            flux_y_A[i, j] = Ĥ₁
            flux_y_Qx[i, j] = Ĥ₂
            flux_y_Qy[i, j] = Ĥ₃
        end
    end

    @inbounds for i in 1:Nx
        for j in 1:Ny
            duA[i, j] = -((flux_x_A[i+1, j] - flux_x_A[i, j]) / Δx +
                          (flux_y_A[i, j+1] - flux_y_A[i, j]) / Δy)
            duQx[i, j] = -((flux_x_Qx[i+1, j] - flux_x_Qx[i, j]) / Δx +
                           (flux_y_Qx[i, j+1] - flux_y_Qx[i, j]) / Δy) - p.friction * Qx[i, j] / A[i, j]
            duQy[i, j] = -((flux_x_Qy[i+1, j] - flux_x_Qy[i, j]) / Δx +
                           (flux_y_Qy[i, j+1] - flux_y_Qy[i, j]) / Δy) - p.friction * Qy[i, j] / A[i, j]
        end
    end

    return nothing
end

"""
    main_two(; kwargs...)

Run a 2D simulation of the (A, Qₓ, Qᵧ) system using a dimensionally split
Lax–Friedrichs (Rusanov) finite-volume method in space coupled with SciML's
`ODEProblem` in time.

The function returns a named tuple containing the spatial grid, solution,
and parameters used in the simulation.
"""
function main_two(; Lx=10.0, Ly=5.0, Nx=80, Ny=40, tspan=(0.0, 0.5), A₀=1.0, β=2.0e5,
                 ρ=1060.0, friction=50.0, inflow_velocity=0.1, saveat=0.001, plot_solution=true)
    Δx = Lx / Nx
    Δy = Ly / Ny
    x = range(Δx/2, length=Nx, step=Δx)
    y = range(Δy/2, length=Ny, step=Δy)

    A_left(t, yval) = A₀ * (1 + 0.05 * sin(2π * t) * exp(-((yval - Ly/2)^2) / (0.3 * Ly)^2))
    Qx_left(t, yval) = inflow_velocity * A_left(t, yval)
    Qy_left(t, yval) = 0.0

    A_right(t, yval) = A₀
    Qx_right(t, yval) = inflow_velocity * A₀
    Qy_right(t, yval) = 0.0

    A_bottom(t, xval) = A₀
    Qx_bottom(t, xval) = inflow_velocity * A₀
    Qy_bottom(t, xval) = 0.0

    A_top(t, xval) = A₀
    Qx_top(t, xval) = inflow_velocity * A₀
    Qy_top(t, xval) = 0.0

    params = AQ2DParams(Nx, Ny, Δx, Δy, A₀, β, ρ, friction, x, y,
                        A_left, Qx_left, Qy_left,
                        A_right, Qx_right, Qy_right,
                        A_bottom, Qx_bottom, Qy_bottom,
                        A_top, Qx_top, Qy_top)

    u0 = zeros(3 * Nx * Ny)
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i
            xc, yc = x[i], y[j]
            perturb = 0.05 * exp(-((xc - Lx / 2)^2 + (yc - Ly / 2)^2) / (0.2 * max(Lx, Ly))^2)
            A_val = A₀ * (1 + perturb)
            u0[3idx - 2] = A_val
            u0[3idx - 1] = inflow_velocity * A_val
            u0[3idx] = 0.0
        end
    end

    prob = ODEProblem(aq_rhs_2d!, u0, tspan, params)
    sol = solve(prob, Tsit5(); saveat=saveat)

    if plot_solution
        plt = plot(layout = (1, 3), size = (1200, 400))
        last_state = sol.u[end]
        A_final = reshape(@view(last_state[1:3:end]), Nx, Ny)
        Qx_final = reshape(@view(last_state[2:3:end]), Nx, Ny)
        Qy_final = reshape(@view(last_state[3:3:end]), Nx, Ny)

        heatmap!(plt[1], x, y, A_final'; aspect_ratio = Ly / Lx,
                 xlabel = "x", ylabel = "y", title = "Area A (cm²)")
        heatmap!(plt[2], x, y, Qx_final'; aspect_ratio = Ly / Lx,
                 xlabel = "x", ylabel = "y", title = "Flow Qₓ (cm³/s)")
        heatmap!(plt[3], x, y, Qy_final'; aspect_ratio = Ly / Lx,
                 xlabel = "x", ylabel = "y", title = "Flow Qᵧ (cm³/s)")
        display(plt)
    end

    return (; x, y, sol, params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result_1d = main_one(plot_solution=false)
    result_2d = main_two(plot_solution=false)
    @info "1D and 2D simulations complete" result_1d result_2d
end
