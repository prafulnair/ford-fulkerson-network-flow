"""Core data structures representing graphs and residual networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]


@dataclass(frozen=True)
class GraphInstance:
    """Immutable snapshot of a generated graph and its metadata."""

    vertices: Sequence[Vertex]
    edges: Iterable[Edge]
    capacities: Mapping[Edge, int]
    adjacency_list: Mapping[Vertex, Sequence[Vertex]]
    source: Vertex
    sink: Vertex
    n: int
    r: float
    upper_cap: float
    max_distance: float | None = None
    total_edges: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(self.vertices))
        object.__setattr__(self, "edges", frozenset(self.edges))
        object.__setattr__(
            self,
            "adjacency_list",
            {vertex: tuple(neighbors) for vertex, neighbors in self.adjacency_list.items()},
        )
        object.__setattr__(self, "capacities", dict(self.capacities))
        if self.total_edges is None:
            object.__setattr__(self, "total_edges", len(self.edges))

    def create_residual_network(self) -> "ResidualNetwork":
        """Construct a fresh residual network derived from this graph."""

        return ResidualNetwork(graph=self)


@dataclass
class ResidualNetwork:
    """Mutable view over residual capacities derived from a graph instance."""

    graph: GraphInstance
    capacities: MutableMapping[Edge, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capacities:
            self.capacities = dict(self.graph.capacities)
        else:
            self.capacities = dict(self.capacities)

    @property
    def source(self) -> Vertex:
        return self.graph.source

    @property
    def sink(self) -> Vertex:
        return self.graph.sink

    @property
    def vertices(self) -> Tuple[Vertex, ...]:
        return self.graph.vertices

    def get_capacity(self, u: Vertex, v: Vertex) -> float:
        """Return the current residual capacity of edge (u, v)."""

        return self.capacities.get((u, v), 0.0)

    def set_capacity(self, u: Vertex, v: Vertex, value: float) -> None:
        """Set the residual capacity on edge (u, v)."""

        if value <= 0:
            self.capacities.pop((u, v), None)
        else:
            self.capacities[(u, v)] = value

    def update(self, u: Vertex, v: Vertex, delta: float) -> float:
        """Adjust the capacity of edge (u, v) by ``delta`` and return the new value."""

        new_value = self.get_capacity(u, v) + delta
        self.set_capacity(u, v, new_value)
        return self.get_capacity(u, v)

    def augment(self, path: Sequence[Edge], bottleneck: float) -> None:
        """Apply an augmentation along ``path`` with the given ``bottleneck`` value."""

        for u, v in path:
            self.update(u, v, -bottleneck)
            self.update(v, u, bottleneck)

    def neighbors(self, u: Vertex) -> List[Vertex]:
        """Return all vertices reachable from ``u`` via positive residual capacity."""

        return [v for (src, v), cap in self.capacities.items() if src == u and cap > 0]

    def max_capacity(self) -> float:
        """Return the maximum residual capacity currently available."""

        positive_capacities = [cap for cap in self.capacities.values() if cap > 0]
        return max(positive_capacities, default=0.0)

    def clone(self) -> "ResidualNetwork":
        """Return a deep copy of this residual network."""

        return ResidualNetwork(graph=self.graph, capacities=self.capacities)

    def snapshot(self) -> "ResidualNetwork":
        """Alias for :meth:`clone` to comply with various naming preferences."""

        return self.clone()

