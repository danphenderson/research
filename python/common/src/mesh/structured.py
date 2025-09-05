"""Structured (rectilinear) meshes in 1‑D and 2‑D.

These meshes store coordinates explicitly so that they can be transferred to
any backend (NumPy, CuPy, PyTorch, JAX‑NumPy, …) by invoking
``mesh.coordinates(backend)``.  All heavy lifting is done once at
construction time using NumPy; subsequent calls cast to the requested backend.

NOTE: 3‑D can be added analogously.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.mesh.base import (  # type: ignore  # imported for protocol compliance
    Array,
    Mesh,
)


class _StructuredMeshBase(Mesh):
    """Utility base: common helpers & tag storage."""

    def __init__(self) -> None:
        self._tags: Dict[Tuple[int, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Tagging helpers ---------------------------------------------------
    # ------------------------------------------------------------------
    def set_entity_tag(self, dim: int, tag: str, indices: List[int]) -> None:  # type: ignore[override]
        self._tags[(dim, tag)] = np.asarray(indices, dtype=int)

    def get_entity_tag(self, dim: int, tag: str) -> np.ndarray:  # type: ignore[override]
        return self._tags.get((dim, tag), np.empty(0, dtype=int))

    # ------------------------------------------------------------------
    # Refinement / partitioning (naïve placeholders) --------------------
    # ------------------------------------------------------------------
    def refine(self, elements: Optional[Sequence[int]] = None) -> "Mesh":  # type: ignore[override]
        """Return a globally refined mesh (split every cell uniformly).

        *1‑D*: each interval is split in two.
        *2‑D*: each quad is split into four (x‑mid & y‑mid).
        For brevity only global refinement is implemented now.
        """
        raise NotImplementedError("Refinement not yet implemented for StructuredMesh.")

    def partition(self, nparts: int) -> Sequence["Mesh"]:  # type: ignore[override]
        raise NotImplementedError(
            "Partitioning not yet implemented for StructuredMesh."
        )


class StructuredMesh1D(_StructuredMeshBase):
    """Uniform line mesh [xmin, xmax] with *nx* nodes (nx ≥ 2)."""

    def __init__(self, xmin: float, xmax: float, nx: int):
        super().__init__()
        if nx < 2:
            raise ValueError("nx must be ≥ 2 for a 1‑D mesh")
        self._xmin, self._xmax, self._nx = float(xmin), float(xmax), int(nx)
        # Coordinates & connectivity in NumPy (host). ---------------------------------
        self._coords_np: np.ndarray = np.linspace(xmin, xmax, nx).reshape(
            -1, 1
        )  # (N,1)
        self._conn_np: np.ndarray = np.vstack(
            [np.arange(nx - 1), np.arange(1, nx)]
        ).T  # (nelem, 2)

    # ------------------------------------------------------------------
    # Mesh interface ----------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return 1

    def coordinates(self, backend: Any = np) -> Array:  # type: ignore[override]
        return backend.array(self._coords_np) if backend is not np else self._coords_np

    def connectivity(self) -> Sequence[Sequence[int]]:  # type: ignore[override]
        return self._conn_np.tolist()

    @property
    def num_nodes(self) -> int:  # type: ignore[override]
        return self._nx

    @property
    def num_elements(self) -> int:  # type: ignore[override]
        return self._nx - 1

    def boundary_nodes(self) -> Sequence[int]:  # type: ignore[override]
        return [0, self._nx - 1]

    # Simple adjacency: vertex‑to‑cell and cell‑to‑vertex ----------------
    def adjacency(self, dim_from: int, dim_to: int) -> Sequence[Sequence[int]]:  # type: ignore[override]
        if dim_from == 0 and dim_to == 1:  # nodes → cells
            adj = [[] for _ in range(self.num_nodes)]
            for e, (a, b) in enumerate(self._conn_np):
                adj[a].append(e)
                adj[b].append(e)
            return adj
        if dim_from == 1 and dim_to == 0:  # cells → nodes
            return self._conn_np.tolist()
        raise NotImplementedError(
            "Only (0→1) and (1→0) adjacencies provided for 1‑D mesh."
        )


class StructuredMesh2D(_StructuredMeshBase):
    """Tensor‑product grid on [xmin,xmax]×[ymin,ymax] with nx×ny nodes."""

    def __init__(
        self, xmin: float, xmax: float, nx: int, ymin: float, ymax: float, ny: int
    ):
        super().__init__()
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny must be ≥ 2 for a 2‑D mesh")
        self._xmin, self._xmax, self._nx = float(xmin), float(xmax), int(nx)
        self._ymin, self._ymax, self._ny = float(ymin), float(ymax), int(ny)
        # Coordinates ----------------------------------------------------------------
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        xv, yv = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)
        self._coords_np = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (N,2)
        # Connectivity: quads (i,j) → 4 vertices --------------------------------------
        conn_list: List[Tuple[int, int, int, int]] = []

        def idx(i: int, j: int) -> int:
            return i * ny + j

        for i in range(nx - 1):
            for j in range(ny - 1):
                v0 = idx(i, j)
                v1 = idx(i + 1, j)
                v2 = idx(i + 1, j + 1)
                v3 = idx(i, j + 1)
                conn_list.append((v0, v1, v2, v3))
        self._conn_np = np.asarray(conn_list, dtype=int)

    # ------------------------------------------------------------------
    # Mesh interface ----------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return 2

    def coordinates(self, backend: Any = np) -> Array:  # type: ignore[override]
        return backend.array(self._coords_np) if backend is not np else self._coords_np

    def connectivity(self) -> Sequence[Sequence[int]]:  # type: ignore[override]
        return self._conn_np.tolist()

    @property
    def num_nodes(self) -> int:  # type: ignore[override]
        return self._nx * self._ny

    @property
    def num_elements(self) -> int:  # type: ignore[override]
        return (self._nx - 1) * (self._ny - 1)

    def boundary_nodes(self) -> Sequence[int]:  # type: ignore[override]
        boundary: List[int] = []
        for i in range(self._nx):
            for j in range(self._ny):
                if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                    boundary.append(i * self._ny + j)
        return boundary

    # Adjacency helpers -------------------------------------------------
    def adjacency(self, dim_from: int, dim_to: int) -> Sequence[Sequence[int]]:  # type: ignore[override]
        if dim_from == 0 and dim_to == 2:  # node → cell (quads)
            adj: List[List[int]] = [[] for _ in range(self.num_nodes)]
            for e, verts in enumerate(self._conn_np):
                for v in verts:
                    adj[v].append(e)
            return adj
        if dim_from == 2 and dim_to == 0:  # cell → node
            return self._conn_np.tolist()
        raise NotImplementedError(
            "Only (0→2) and (2→0) adjacencies provided for 2‑D mesh."
        )
