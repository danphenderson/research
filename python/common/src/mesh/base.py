# xde/mesh/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

Array = Any


class Mesh(ABC):
    """Static mesh representation (nodes, elements, etc.)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the mesh dimension (e.g. 1, 2, or 3)."""

    @abstractmethod
    def coordinates(self, backend: Any) -> Array:
        """Return node coordinates as an array of shape (N, dimension)."""

    @abstractmethod
    def connectivity(self) -> Sequence[Sequence[int]]:
        """Return element connectivity (list of node-index lists)."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes in the mesh."""

    @property
    @abstractmethod
    def num_elements(self) -> int:
        """Number of elements (cells) in the mesh."""

    def adjacency(self, dim_from: int, dim_to: int) -> Sequence[Sequence[int]]:
        """Return adjacency information between entities of different dimensions."""
        raise NotImplementedError("Adjacency information is not implemented.")

    @abstractmethod
    def boundary_nodes(self) -> Sequence[int]:
        """Return indices of nodes on the domain boundary."""

    @abstractmethod
    def set_entity_tag(self, dim: int, tag: str, indices: List[int]) -> None:
        """Set a tag for a set of entities (nodes, edges, faces)."""

    @abstractmethod
    def refine(self, elements: Optional[Sequence[int]] = None) -> "Mesh":
        """Refine the mesh by splitting elements into smaller ones."""

    @abstractmethod
    def partition(self, nparts: int) -> Sequence["Mesh"]:
        """Partition the mesh into `nparts` sub-meshes."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension}, num_nodes={self.num_nodes}, num_elements={self.num_elements})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {self.num_nodes} nodes and {self.num_elements} elements."

    def __getitem__(self, item: int) -> Any:
        """Return the node or element at the specified index."""
        if item < 0:
            item += self.num_nodes
        if item < 0 or item >= self.num_nodes:
            raise IndexError("Index out of range.")
        return self.coordinates()[item]

    def __len__(self) -> int:
        """Return the number of nodes in the mesh."""
        return self.num_nodes

    def __iter__(self) -> Any:
        """Return an iterator over the nodes in the mesh."""
        return iter(self.coordinates())

    def __contains__(self, item: int) -> bool:
        """Check if the specified node index is in the mesh."""
        if item < 0:
            item += self.num_nodes
        return 0 <= item < self.num_nodes

    def __eq__(self, other: Any) -> bool:
        """Check if two meshes are equal based on their properties."""
        if not isinstance(other, Mesh):
            return False
        return (
            self.dimension == other.dimension
            and self.num_nodes == other.num_nodes
            and self.num_elements == other.num_elements
            and np.array_equal(self.coordinates(), other.coordinates())
            and self.connectivity() == other.connectivity()
        )

    def __ne__(self, other: Any) -> bool:
        """Check if two meshes are not equal."""
        return not self.__eq__(other)
