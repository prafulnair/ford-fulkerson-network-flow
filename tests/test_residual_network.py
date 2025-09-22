"""Unit tests for ``ResidualNetwork`` helpers."""

from __future__ import annotations

import pytest

from ford_fulkerson.models import GraphInstance, Vertex


def _single_edge_path(graph: GraphInstance) -> list[tuple[Vertex, Vertex]]:
    return [(graph.source, graph.sink)]


def test_augment_updates_forward_and_reverse(single_edge_case) -> None:
    """Augmenting along a path reduces forward capacity and exposes reverse flow."""

    graph, vertices = single_edge_case
    network = graph.create_residual_network()

    path = _single_edge_path(graph)
    network.augment(path, bottleneck=3)

    assert network.get_capacity(vertices["s"], vertices["t"]) == pytest.approx(4)
    assert network.get_capacity(vertices["t"], vertices["s"]) == pytest.approx(3)

    network.augment(path, bottleneck=4)
    assert network.get_capacity(vertices["s"], vertices["t"]) == 0
    assert network.get_capacity(vertices["t"], vertices["s"]) == pytest.approx(7)


def test_clone_returns_independent_copy(parallel_paths_case) -> None:
    """Mutating the original residual network does not leak into clones."""

    graph, vertices = parallel_paths_case
    original = graph.create_residual_network()
    clone = original.clone()

    path = [(vertices["s"], vertices["a"]), (vertices["a"], vertices["t"])]
    original.augment(path, bottleneck=1)

    assert original.get_capacity(vertices["s"], vertices["a"]) == pytest.approx(1)
    assert original.get_capacity(vertices["a"], vertices["t"]) == pytest.approx(1)
    assert clone.get_capacity(vertices["s"], vertices["a"]) == pytest.approx(2)
    assert clone.get_capacity(vertices["a"], vertices["t"]) == pytest.approx(2)
    assert clone.get_capacity(vertices["t"], vertices["a"]) == 0

    # Cloning should preserve the underlying graph reference so metadata stays shared.
    assert clone.graph is original.graph
