import matplotlib.pyplot as plt
import numpy as np


def solve_upwind_1D(
    a_func, u0_func, Nx=200, xmin=0.0, xmax=1.0, CFL=0.8, T_final=1.0, save_intervals=5
):
    """
    Solves the linear advection equation u_t + a(x) * u_x = 0 using the explicit upwind scheme.

    Parameters:
        a_func        : function a(x), velocity field
        u0_func       : function u(x,0), initial condition
        Nx            : number of spatial cells
        xmin, xmax    : spatial domain limits
        CFL           : Courant–Friedrichs–Lewy number
        T_final       : final time
        save_intervals: number of time snapshots to plot
    """
    dx = (xmax - xmin) / Nx
    x_grid = np.linspace(xmin, xmax, Nx + 1)

    u = u0_func(x_grid)
    max_speed = np.max(np.abs(a_func(x_grid)))
    dt = CFL * dx / max_speed

    num_steps = int(np.ceil(T_final / dt))
    dt = T_final / num_steps  # Adjust dt to exactly reach T_final

    times_to_save = np.linspace(0, num_steps, save_intervals + 1, dtype=int)
    solutions = [u.copy()]
    time_points = [0.0]

    for n in range(1, num_steps + 1):
        u_new = u.copy()

        for j in range(1, Nx):
            aj = a_func(x_grid[j])
            if aj >= 0:
                u_new[j] = u[j] - (aj * dt / dx) * (u[j] - u[j - 1])
            else:
                u_new[j] = u[j] - (aj * dt / dx) * (u[j + 1] - u[j])

        # Left boundary (inflow if aj < 0)
        aj_left = a_func(x_grid[0])
        if aj_left > 0:
            u_new[0] = u[0]

        # Right boundary (inflow if aj > 0)
        aj_right = a_func(x_grid[Nx])
        if aj_right < 0:
            u_new[Nx] = u[Nx]

        u = u_new

        if n in times_to_save:
            solutions.append(u.copy())
            time_points.append(n * dt)

    plt.figure(figsize=(8, 5))
    for sol, t in zip(solutions, time_points):
        plt.plot(x_grid, sol, label=f"t = {t:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Upwind Scheme: $u_t + a(x) u_x = 0$")
    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def solve_upwind_2d_3d_animation(
    a_func,
    b_func,
    u0_func,
    Nx=100,
    Ny=100,
    xmin=0.0,
    xmax=1.0,
    ymin=0.0,
    ymax=1.0,
    CFL=0.4,
    T_final=1.0,
    save_frames=100,
):
    """
    Solves the 2D linear advection equation:
        u_t + a(x,y) u_x + b(x,y) u_y = 0
    using an explicit upwind scheme.

    Parameters:
        a_func        : function a(x,y), x-direction velocity
        b_func        : function b(x,y), y-direction velocity
        u0_func       : function u(x,y), initial condition
        Nx, Ny        : number of grid points in x and y
        xmin, xmax    : spatial domain in x
        ymin, ymax    : spatial domain in y
        CFL           : CFL number
        T_final       : final simulation time
        save_intervals: number of snapshots to plot
    """
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

    x = np.linspace(xmin, xmax, Nx + 1)
    y = np.linspace(ymin, ymax, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = u0_func(X, Y)
    a_vals = a_func(X, Y)
    b_vals = b_func(X, Y)

    max_speed = np.max(np.abs(a_vals)) / dx + np.max(np.abs(b_vals)) / dy
    dt = CFL / max_speed
    num_steps = int(np.ceil(T_final / dt))
    dt = T_final / num_steps

    step_interval = max(1, num_steps // save_frames)
    frames = []
    times = []

    for n in range(1, num_steps + 1):
        u_new = u.copy()

        for i in range(1, Nx):
            for j in range(1, Ny):
                ax = a_vals[i, j]
                bx = b_vals[i, j]

                dx_term = (
                    (u[i, j] - u[i - 1, j]) / dx
                    if ax >= 0
                    else (u[i + 1, j] - u[i, j]) / dx
                )
                dy_term = (
                    (u[i, j] - u[i, j - 1]) / dy
                    if bx >= 0
                    else (u[i, j + 1] - u[i, j]) / dy
                )

                u_new[i, j] = u[i, j] - dt * (ax * dx_term + bx * dy_term)

        u = u_new

        if n % step_interval == 0:
            frames.append(u.copy())
            times.append(n * dt)

    # Animation (3D surface)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = [ax.plot_surface(X, Y, frames[0], cmap="viridis")]

    def update_surface(frame_idx):
        ax.clear()
        ax.set_title(f"t = {times[frame_idx]:.2f}")
        ax.set_zlim(0, np.max(frames[0]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x, y, t)")
        return ax.plot_surface(X, Y, frames[frame_idx], cmap="viridis")

    ani = animation.FuncAnimation(
        fig, update_surface, frames=len(frames), interval=50, blit=False
    )

    ani.save("advection_3d.gif", writer="pillow", fps=20)
    print("Animation saved as advection_3d.gif")
    plt.close()


# def a_func(x):
#     return 1.0
#
# def u0_parabola(x):
#     return x * (1 - x)
#
# solve_upwind_1D(a_func=a_func, u0_func=u0_parabola)


def a_const(x, y):
    return 1.0 * np.ones_like(x)


def b_const(x, y):
    return 0.5 * np.ones_like(x)


def u0_gaussian(x, y):
    return np.exp(-100 * ((x - 0.3) ** 2 + (y - 0.3) ** 2))


solve_upwind_2d_3d_animation(
    a_func=a_const, b_func=b_const, u0_func=u0_gaussian, T_final=1.0
)
