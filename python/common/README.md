# `XDE`

Key goals and requirements of package: - Mesh Management: Abstract mesh representation (structured/unstructured, 1D/2D/3D) with methods for connectivity, refinement, and tagging boundary entities. - Boundary Conditions: Separate handling of Dirichlet, Neumann, etc., allowing specification via functions or objects. - PDE Operator: A unified interface for spatial discretization that different methods (FEM/DG/FDM) implement (Strategy). - Time Stepping: Pluggable integrators (Euler, Runge–Kutta, etc.) that use the operator to advance solutions. - Solver Facade: A high-level Solver class that orchestrates mesh, operator, BCs, and time-stepping to run a simulation (Facade). - CPU/GPU Support: Abstract array/matrix backends so code can run on NumPy or CuPy (Python) and on Julia’s CPU/GPU arrays, chosen via configuration.
Below we outline each core component and its interface,
then discuss patterns and principles that guide the design.

## Core Components

### Abstract Interfaces

1. **Backend** declares an interface to support both CPU/GPU hardware backends,
   the operator can allocate and operate on generic array types. This decouples
   the operator from the underlying array library (NumPy, PyTorch, etc.),
   the PDE operators and steppers simply call backend.add, backend.matmul, etc.

```python
class Backend:
    """Abstract backend interface."""
    @abstractmethod
    def array(self, data, dtype=None): ...
    @abstractmethod
    def zeros(self, shape): ...
    @abstractmethod
    def ones(self, shape): ...
    @abstractmethod
    def linspace(self, a, b, num): ...
    @abstractmethod
    def add(self, x, y): ...
    @abstractmethod
    def matmul(self, A, B): ...
    # ... other needed operations
```

This layer adheres to the Dependency Inversion principle: high-level code (operators, solvers) depends on the abstraction ArrayBackend,
not on specific libraries. New backends (e.g. JAX, CuPy) could be added by implementing the same interface.

2. **Mesh** interface encapsulates the computational domain geometry.
   Concrete mesh classes (e.g. StructuredMesh, UnstructuredMesh) implement
   the following interface contract:

```python
class Mesh(ABC):
    @abstractmethod
    def num_nodes(self) -> int: ...
    @abstractmethod
    def coordinates(self): ...
    @abstractmethod
    def elements(self): ...
    @abstractmethod
    def boundary_nodes(self): ...
```

This enables solvers and operators to depend only on the Mesh abstraction, not on mesh implementation details

3. **Boundary conditions** (BCs) inheret from a common interface.

```python
class BoundaryConditions:
    """Abstract boundary condition interface."""
    def apply(self, mesh: Mesh, vector: Array): ...
```

Concrete BC types (DirichletBC, NeumannBC, etc.) implement `apply()` such that:

-   Dirichlet conditions set values on the boundary nodes
-   Neumann conditions set values on the boundary faces and numerical fluxes; to enforce conservation of mass
-   Mixed/Robin conditions can be a combination of Dirichlet and Neumann

4. **PDEOperator** encapsulates the spatial discretization of the PDE
   (the right-hand-side or residual). An operator takes the current state
   (field values on the mesh) and returns its time derivative or residual.
   Example interface:

```python
class PDEOperator(ABC):
    @abstractmethod
    def apply(self, mesh: Mesh, u, t: float=None):
        """Compute PDE right-hand side or residual on mesh given state u."""
    @abstractmethod
    def dimension(self) -> int: ...
```

Concrete implementations of `PDEOperator` implement the `apply()` method

-   `FDMOperator` implements finite-difference stencils on a StructuredMesh.
-   `FEMOperator` assembles finite-element matrices/vectors (using e.g. UFL or custom assembly).
-   `DGOperator` implements discontinuous-Galerkin fluxes, etc.
    For explicit time-stepping, apply typically returns du/dt at the given state and time.

6. **TimeStepper** abstract integrator for advancing in time. It takes a state and
   a `PDEOperator` and produces a new state after one (or many) time steps.
   Example interface:

```python
class TimeStepper:
    @abstractmethod
    def step(self, mesh: Mesh, u, operator: PDEOperator, dt: float):
        """Advance state u by time dt using the PDEOperator."""
    @abstractmethod
    def integrate(self, mesh: Mesh, u0, operator: PDEOperator, t0: float, t1: float, dt: float):
        """Integrate from t0 to t1 starting at u0, returning final state."""
```

Concrete `TimeStepper` implementation examples include `EulerStepper`,
`SSPRK2Stepper`, `SSPRK3Stepper`, etc. Each `step()` calls
the operator and applies BCs appropriately.

7. **Solver** orchestrates the entire simulation process. It takes a mesh, PDE operator,
   boundary conditions, and time-stepping method as input.
   Example interface:

```python
class Solver:
    def __init__(self, mesh: Mesh, operator: PDEOperator, bcs: List[BoundaryCondition],
                 time_stepper: TimeStepper):
        ...
    def solve(self, u0: Array, t_final: float, dt: float) -> Array:
        """Run time-stepping from t=0 to t_final."""
        ...
```

## Design Patterns and Principles

The design of the library is guided by several software design patterns and principles:

-   **Dependency Inversion Principle**: High-level modules (solvers, operators) depend on abstractions (interfaces) rather than concrete implementations. This allows for easy swapping of components.
-   **Strategy Pattern**: Used for the PDEOperator and BoundaryCondition interfaces. This allows different implementations to be swapped easily.
-   **Factory/Abstract Factory**: Used to create components. E.g., a SolverFactory could construct a solver given a configuration dictionary or parameters, instantiating the appropriate mesh, operator, etc.
-   **Facade**: The Solver class itself is a facade that provides a simple solve() method. It delegates to lower-level subsystems (mesh, operator, linear algebra) while hiding their complexity.
-   **Composition over Inheritance**: While we use inheritance (e.g., base classes for time steppers or operators), much reuse is achieved by composing objects rather than deep class hierarchies.
-   **SOLID Principles**: The library design adheres to SOLID for maintainability. Classes have single responsibilities (e.g., mesh only handles geometry, BC classes only apply conditions). Code is open for extension but closed for modification: new solver methods or BC types can be added without changing existing code. Interfaces are kept minimal (Interface Segregation): for instance, the Mesh interface only includes necessary queries, so clients aren’t forced to depend on unused methods. We also invert dependencies by programming to interfaces (abstract classes or types) rather than concrete classes.
