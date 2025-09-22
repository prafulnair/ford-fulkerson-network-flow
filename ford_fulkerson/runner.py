"""Utilities for executing Ford-Fulkerson strategies over stored graphs."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from math import hypot, isinf
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

from .algorithms import (
    ford_fulkerson,
    ford_fulkerson_DFS_like,
    ford_fulkerson_max_capacity,
    ford_fulkerson_random,
)
from .io import read_data
from .models import Edge, GraphInstance, ResidualNetwork, Vertex

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


def _read_vertices(path: str) -> Sequence[Vertex]:
    vertices: list[Vertex] = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            vertices.append((float(row[0]), float(row[1])))
    return vertices


def _read_edges(path: str) -> Sequence[Edge]:
    edges: list[Edge] = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            edges.append((u, v))
    return edges


def _read_capacities(path: str) -> Dict[Edge, int]:
    capacities: Dict[Edge, int] = {}
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            capacities[(u, v)] = int(float(row[4]))
    return capacities


def _read_adjacency(path: str) -> Dict[Vertex, Tuple[Vertex, ...]]:
    adjacency: Dict[Vertex, Tuple[Vertex, ...]] = {}
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            vertex = (float(row[0]), float(row[1]))
            neighbors: list[Vertex] = []
            for index in range(2, len(row), 2):
                if row[index] == "" or index + 1 >= len(row):
                    continue
                neighbors.append((float(row[index]), float(row[index + 1])))
            adjacency[vertex] = tuple(neighbors)
    return adjacency


def _guess_source(adjacency: Mapping[Vertex, Sequence[Vertex]], vertex_order: Mapping[Vertex, int]) -> Vertex:
    indegree: Dict[Vertex, int] = {vertex: 0 for vertex in vertex_order}
    for neighbours in adjacency.values():
        for head in neighbours:
            indegree[head] = indegree.get(head, 0) + 1

    def _score(vertex: Vertex) -> Tuple[int, int, int]:
        out_degree = len(adjacency.get(vertex, ()))
        diff = out_degree - indegree.get(vertex, 0)
        # Prefer higher diff, then higher out-degree, then stable tie-breaking by
        # the original vertex order.
        return (diff, out_degree, -vertex_order[vertex])

    return max(vertex_order, key=_score)


def _longest_distance_from(
    source: Vertex, adjacency: Mapping[Vertex, Sequence[Vertex]]
) -> Tuple[Vertex, float]:
    distances: Dict[Vertex, float] = {vertex: float("inf") for vertex in adjacency}
    distances[source] = 0

    queue: deque[Vertex] = deque([source])
    while queue:
        tail = queue.popleft()
        for head in adjacency.get(tail, ()):  # pragma: no branch - defensive lookup
            if distances[head] != float("inf"):
                continue
            distances[head] = distances[tail] + 1
            queue.append(head)

    sink = source
    max_distance = 0.0
    for vertex, distance in distances.items():
        if distance != float("inf") and distance > max_distance:
            max_distance = distance
            sink = vertex

    return sink, max_distance


def _reconstruct_graph(
    graph_no: int,
    vertices_path: str,
    edges_path: str,
    capacities_path: str,
    adjacency_path: str,
) -> GraphInstance:
    vertices = _read_vertices(vertices_path)
    edges = _read_edges(edges_path)
    capacities = _read_capacities(capacities_path)
    adjacency = _read_adjacency(adjacency_path)

    if not vertices:
        raise FileNotFoundError(f"No vertices available for graph {graph_no}")

    vertex_order = {vertex: index for index, vertex in enumerate(vertices)}
    source = _guess_source(adjacency, vertex_order)
    sink, max_distance = _longest_distance_from(source, adjacency)

    max_edge_length = max(
        (hypot(head[0] - tail[0], head[1] - tail[1]) for tail, head in edges),
        default=0.0,
    )
    upper_cap = max(capacities.values(), default=0)

    print(
        "sim_val_{graph_no}_meta_info.csv missing; reconstructed metadata from graph".format(
            graph_no=graph_no
        )
    )

    return GraphInstance(
        vertices=vertices,
        edges=edges,
        capacities=capacities,
        adjacency_list={vertex: list(neighbours) for vertex, neighbours in adjacency.items()},
        source=source,
        sink=sink,
        n=len(vertices),
        r=max_edge_length,
        upper_cap=upper_cap,
        max_distance=max_distance,
        total_edges=len(edges),
    )


def _load_graph(graph_no: int) -> GraphInstance:
    paths = _graph_file_paths(graph_no)
    meta_path = Path(paths[0])
    if meta_path.exists():
        return read_data(*paths)

    return _reconstruct_graph(graph_no, *paths[1:])


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

    graph = _load_graph(graph_no)
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

