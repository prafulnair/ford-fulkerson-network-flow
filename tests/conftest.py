"""Shared fixtures for deterministic residual network test cases."""

from __future__ import annotations

import sys
import types
from typing import Dict, Mapping, Sequence, Tuple

import pytest


def _install_matplotlib_stub() -> None:
    """Provide a minimal ``matplotlib`` stub when the dependency is missing."""

    if "matplotlib" in sys.modules:
        return

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("pyplot")

    def _missing(*_args, **_kwargs):  # pragma: no cover - defensive stub
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualisation; install it to use this feature."
        )

    for name in ("scatter", "arrow", "show", "xlabel", "ylabel", "title", "legend"):
        setattr(pyplot, name, _missing)

    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot


_install_matplotlib_stub()

from ford_fulkerson.models import Edge, GraphInstance, Vertex


def _build_graph(
    vertices: Mapping[str, Vertex],
    edge_order: Sequence[Tuple[str, str]],
    capacities: Mapping[Tuple[str, str], int],
    source: str,
    sink: str,
) -> GraphInstance:
    """Create a ``GraphInstance`` with adjacency derived from ``edge_order``."""

    adjacency_list: Dict[Vertex, list[Vertex]] = {coord: [] for coord in vertices.values()}
    typed_capacities: Dict[Edge, int] = {}

    # Python preserves insertion order for dicts/lists which ensures deterministic
    # neighbour traversal across the tests when the algorithms iterate over
    # ``capacities`` and ``adjacency_list``.
    edge_sequence: list[Edge] = []
    for tail, head in edge_order:
        u = vertices[tail]
        v = vertices[head]
        edge = (u, v)
        adjacency_list[u].append(v)
        edge_sequence.append(edge)
        typed_capacities[edge] = capacities[(tail, head)]

    return GraphInstance(
        vertices=tuple(vertices.values()),
        edges=tuple(edge_sequence),
        capacities=typed_capacities,
        adjacency_list=adjacency_list,
        source=vertices[source],
        sink=vertices[sink],
        n=len(vertices),
        r=0.0,
        upper_cap=float(max(capacities.values(), default=0)),
        max_distance=0,
        total_edges=len(edge_sequence),
    )


@pytest.fixture
def single_edge_case() -> Tuple[GraphInstance, Dict[str, Vertex]]:
    """Graph with one direct edge from source to sink (capacity 7)."""

    vertices: Dict[str, Vertex] = {
        "s": (0.0, 0.0),
        "t": (1.0, 0.0),
    }
    edges = [("s", "t")]
    capacities = {("s", "t"): 7}
    graph = _build_graph(vertices, edges, capacities, source="s", sink="t")
    return graph, vertices


@pytest.fixture
def parallel_paths_case() -> Tuple[GraphInstance, Dict[str, Vertex]]:
    """Graph with two disjoint unit-length paths delivering total flow three."""

    vertices: Dict[str, Vertex] = {
        "s": (0.0, 0.0),
        "a": (0.5, 0.0),
        "b": (0.5, 0.5),
        "t": (1.0, 0.0),
    }
    edges = [
        ("s", "a"),
        ("a", "t"),
        ("s", "b"),
        ("b", "t"),
    ]
    capacities = {
        ("s", "a"): 2,
        ("a", "t"): 2,
        ("s", "b"): 1,
        ("b", "t"): 1,
    }
    graph = _build_graph(vertices, edges, capacities, source="s", sink="t")
    return graph, vertices


@pytest.fixture
def single_edge_graph(single_edge_case: Tuple[GraphInstance, Dict[str, Vertex]]) -> GraphInstance:
    """Expose only the ``GraphInstance`` for strategy-focused tests."""

    graph, _ = single_edge_case
    return graph


@pytest.fixture
def parallel_paths_graph(
    parallel_paths_case: Tuple[GraphInstance, Dict[str, Vertex]]
) -> GraphInstance:
    """Expose only the ``GraphInstance`` for strategy-focused tests."""

    graph, _ = parallel_paths_case
    return graph
