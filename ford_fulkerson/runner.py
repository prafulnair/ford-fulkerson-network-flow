"""Utilities for executing Ford-Fulkerson strategies over stored graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Callable, Dict, Mapping, MutableMapping, Tuple

from .algorithms import (
    ford_fulkerson,
    ford_fulkerson_DFS_like,
    ford_fulkerson_max_capacity,
    ford_fulkerson_random,
)
from .io import read_data
from .models import GraphInstance, ResidualNetwork

StrategyCallable = Callable[[ResidualNetwork], Tuple[float, int, float]]


@dataclass(frozen=True)
class StrategyMetrics:
    """Summary metrics reported by a Ford-Fulkerson strategy."""

    max_flow: float
    augmenting_paths: int
    mean_length: float
    mpl: float


@dataclass(frozen=True)
class GraphRunResult:
    """Container for the metrics gathered from all strategies for a graph."""

    graph_no: int
    graph: GraphInstance
    strategies: Dict[str, StrategyMetrics]


DEFAULT_STRATEGIES: MutableMapping[str, StrategyCallable] = {
    "sap": ford_fulkerson,
    "dfs_like": ford_fulkerson_DFS_like,
    "max_capacity": ford_fulkerson_max_capacity,
    "random": ford_fulkerson_random,
}


def _graph_file_paths(graph_no: int) -> Tuple[str, str, str, str, str]:
    prefix = f"sim_val_{graph_no}"
    return (
        f"{prefix}_meta_info.csv",
        f"{prefix}_vertices.csv",
        f"{prefix}_edges.csv",
        f"{prefix}_capacities.csv",
        f"{prefix}_adjlist.csv",
    )


def _compute_mpl(max_distance: float | None, mean_length: float) -> float:
    if not max_distance or isinf(float(max_distance)):
        return 0.0
    return mean_length / float(max_distance)


def run_strategies_for_graph(
    graph_no: int,
    strategies: Mapping[str, StrategyCallable] | None = None,
) -> GraphRunResult:
    """Load ``graph_no`` once and execute each strategy on fresh clones."""

    strategy_map: Dict[str, StrategyCallable]
    if strategies is None:
        strategy_map = dict(DEFAULT_STRATEGIES)
    else:
        strategy_map = dict(strategies)

    graph = read_data(*_graph_file_paths(graph_no))
    base_network = graph.create_residual_network()

    print(f"\nGraph {graph_no}: source {graph.source}, sink {graph.sink}")
    if graph.total_edges is not None:
        print(f"Total edges: {graph.total_edges}")
    if graph.seed is not None:
        print(f"Seed: {graph.seed}")

    results: Dict[str, StrategyMetrics] = {}

    for strategy_name, strategy in strategy_map.items():
        network = base_network.clone()
        max_flow, augmenting_paths, mean_length = strategy(network)
        mpl = _compute_mpl(graph.max_distance, mean_length)

        metrics = StrategyMetrics(
            max_flow=max_flow,
            augmenting_paths=augmenting_paths,
            mean_length=mean_length,
            mpl=mpl,
        )

        results[strategy_name] = metrics

        print(
            "[{name}] max_flow={max_flow}, augmenting_paths={paths}, "
            "mean_length={mean_length}, mpl={mpl}".format(
                name=strategy_name,
                max_flow=max_flow,
                paths=augmenting_paths,
                mean_length=mean_length,
                mpl=mpl,
            )
        )

    return GraphRunResult(graph_no=graph_no, graph=graph, strategies=results)

