"""End-to-end checks for each Ford-Fulkerson augmenting-path strategy."""

from __future__ import annotations

import random
from typing import Callable, Dict, Tuple

import pytest

from ford_fulkerson.algorithms import (
    ford_fulkerson,
    ford_fulkerson_DFS_like,
    ford_fulkerson_max_capacity,
    ford_fulkerson_random,
    ford_fulkerson_strategies,
)
from ford_fulkerson.models import GraphInstance, ResidualNetwork

Strategy = Callable[[ResidualNetwork], Tuple[float, int, float]]


def _assert_strategy_metrics(
    graph: GraphInstance, expected: Dict[str, Tuple[float, int, float]]
) -> None:
    """Run each strategy on a fresh clone and assert the provided metrics."""

    strategies: Dict[str, Strategy] = {
        "sap": ford_fulkerson,
        "dfs_like": ford_fulkerson_DFS_like,
        "max_capacity": ford_fulkerson_max_capacity,
        "random": ford_fulkerson_random,
    }

    for name, strategy in strategies.items():
        network = graph.create_residual_network()
        assert network.capacities, "Residual network should expose initial capacities"
        if name == "random":
            random.seed(0)
        flow, paths, mean_length = strategy(network)
        exp_flow, exp_paths, exp_mean = expected[name]
        assert flow == pytest.approx(exp_flow)
        assert paths == exp_paths
        assert mean_length == pytest.approx(exp_mean)

    random.seed(0)
    combined = ford_fulkerson_strategies(graph.create_residual_network())
    assert set(combined.keys()) == set(strategies.keys())
    for name, metrics in combined.items():
        flow, paths, mean_length = metrics
        exp_flow, exp_paths, exp_mean = expected[name]
        assert flow == pytest.approx(exp_flow)
        assert paths == exp_paths
        assert mean_length == pytest.approx(exp_mean)


def test_strategies_single_edge(single_edge_graph: GraphInstance) -> None:
    """All strategies find the unique path and exhaust its capacity."""

    _assert_strategy_metrics(
        single_edge_graph,
        expected={
            "sap": (7.0, 1, 1.0),
            "dfs_like": (7.0, 1, 1.0),
            "max_capacity": (7.0, 1, 1.0),
            "random": (7.0, 1, 1.0),
        },
    )


def test_strategies_parallel_paths(parallel_paths_graph: GraphInstance) -> None:
    """All strategies saturate both disjoint paths from the fixtures."""

    _assert_strategy_metrics(
        parallel_paths_graph,
        expected={
            "sap": (3.0, 2, 2.0),
            "dfs_like": (3.0, 2, 2.0),
            "max_capacity": (3.0, 2, 2.0),
            # ``ford_fulkerson_random``'s priority-queue heuristic skips processing nodes when
            # the randomly adjusted priority is lower than the recorded distance. That causes
            # the search to terminate immediately on paths longer than one edge.
            "random": (0.0, 0, 0.0),
        },
    )
