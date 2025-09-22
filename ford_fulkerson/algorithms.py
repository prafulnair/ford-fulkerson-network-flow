"""Augmenting path strategies for the Ford-Fulkerson algorithm."""

from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Dict, List, Tuple

from .models import Edge, ResidualNetwork, Vertex


def _reconstruct_path(
    parent: Dict[Vertex, Vertex | None], source: Vertex, sink: Vertex
) -> List[Edge]:
    path: List[Edge] = []
    current = sink

    while current != source:
        predecessor = parent.get(current)
        if predecessor is None:
            return []
        path.append((predecessor, current))
        current = predecessor

    path.reverse()
    return path


def dijkstra_foundation_DFSLike(
    network: ResidualNetwork, parent: Dict[Vertex, Vertex | None], strategy: str
) -> bool:
    """Shared Dijkstra-style search used by DFS-like and random strategies."""

    parent.clear()
    parent[network.source] = None

    distances = {vertex: float("inf") for vertex in network.vertices}
    distances[network.source] = 0
    max_capacity = int(network.max_capacity())
    counter = 0
    priority_queue: List[Tuple[float, Vertex]] = [(0, network.source)]

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)
        if current_distance != distances[u]:
            continue

        for v in network.neighbors(u):
            residual_capacity = network.get_capacity(u, v)
            if residual_capacity <= 0 or distances[v] != float("inf"):
                continue

            distances[v] = distances[u] + 1
            parent[v] = u

            priority_value = distances[v]
            if strategy == "dfs_like":
                priority_value -= counter
                counter -= 1
            elif max_capacity > 0:
                priority_value -= random.randint(0, max_capacity)
                counter -= 1

            heapq.heappush(priority_queue, (priority_value, v))

    return distances[network.sink] != float("inf")


def ford_fulkerson_random(network: ResidualNetwork) -> Tuple[float, int, float]:
    """Ford-Fulkerson using randomized tie-breaking for augmenting paths."""

    parent: Dict[Vertex, Vertex | None] = {}
    max_flow_random = 0.0
    edge_lengths: List[int] = []
    total_augmenting_paths = 0

    while dijkstra_foundation_DFSLike(network, parent, "random"):
        path = _reconstruct_path(parent, network.source, network.sink)
        if not path:
            break
        bottleneck = min(network.get_capacity(u, v) for u, v in path)
        if bottleneck <= 0:
            break

        total_augmenting_paths += 1
        max_flow_random += bottleneck
        edge_lengths.append(len(path))
        network.augment(path, bottleneck)

    mean_length = sum(edge_lengths) / total_augmenting_paths if total_augmenting_paths else 0
    return max_flow_random, total_augmenting_paths, mean_length


def dijkstra_maxCap(network: ResidualNetwork) -> Tuple[float, List[Edge]]:
    priority_queue: List[Tuple[float, Vertex, List[Edge]]] = [
        (-float("inf"), network.source, [])
    ]
    capacity: Dict[Vertex, float] = {vertex: 0.0 for vertex in network.vertices}
    capacity[network.source] = float("inf")

    while priority_queue:
        neg_capacity, current_vertex, current_path = heapq.heappop(priority_queue)
        current_capacity = -neg_capacity

        if current_vertex == network.sink:
            return capacity[current_vertex], current_path

        for neighbor in network.neighbors(current_vertex):
            edge_capacity = network.get_capacity(current_vertex, neighbor)
            if edge_capacity <= 0:
                continue
            min_capacity = min(current_capacity, edge_capacity)
            if min_capacity > capacity.get(neighbor, 0):
                capacity[neighbor] = min_capacity
                new_path = current_path + [(current_vertex, neighbor)]
                heapq.heappush(priority_queue, (-min_capacity, neighbor, new_path))

    return 0.0, []


def ford_fulkerson_max_capacity(network: ResidualNetwork) -> Tuple[float, int, float]:
    total_augmenting_paths = 0
    total_length = 0
    max_flow_max_capacity = 0.0

    while True:
        capacity, augmenting_path = dijkstra_maxCap(network)

        if capacity <= 0 or not augmenting_path:
            break

        total_length += len(augmenting_path)
        total_augmenting_paths += 1
        max_flow_max_capacity += capacity
        network.augment(augmenting_path, capacity)

    mean_length = total_length / total_augmenting_paths if total_augmenting_paths else 0
    return max_flow_max_capacity, total_augmenting_paths, mean_length


def ford_fulkerson_DFS_like(network: ResidualNetwork) -> Tuple[float, int, float]:
    parent: Dict[Vertex, Vertex | None] = {}
    max_flow_DFS_like = 0.0
    edge_lengths: List[int] = []
    total_augmenting_paths = 0

    while dijkstra_foundation_DFSLike(network, parent, "dfs_like"):
        path = _reconstruct_path(parent, network.source, network.sink)
        if not path:
            break
        bottleneck = min(network.get_capacity(u, v) for u, v in path)
        if bottleneck <= 0:
            break

        total_augmenting_paths += 1
        max_flow_DFS_like += bottleneck
        edge_lengths.append(len(path))
        network.augment(path, bottleneck)

    mean_length = sum(edge_lengths) / total_augmenting_paths if total_augmenting_paths else 0
    return max_flow_DFS_like, total_augmenting_paths, mean_length


def BFS_FF_SAP(
    network: ResidualNetwork, parent: Dict[Vertex, Vertex | None]
) -> bool:
    visited = {network.source}
    queue = deque([network.source])

    parent.clear()
    parent[network.source] = None

    while queue:
        u = queue.popleft()

        for v in network.neighbors(u):
            if v in visited:
                continue
            residual = network.get_capacity(u, v)
            if residual <= 0:
                continue
            visited.add(v)
            parent[v] = u
            if v == network.sink:
                return True
            queue.append(v)

    return False


def ford_fulkerson(network: ResidualNetwork) -> Tuple[float, int, float]:
    parent: Dict[Vertex, Vertex | None] = {}
    max_flow_SAP = 0.0
    edge_lengths: List[int] = []
    total_augmenting_paths = 0

    while BFS_FF_SAP(network, parent):
        path = _reconstruct_path(parent, network.source, network.sink)
        if not path:
            break
        bottleneck = min(network.get_capacity(u, v) for u, v in path)
        if bottleneck <= 0:
            break

        total_augmenting_paths += 1
        max_flow_SAP += bottleneck
        edge_lengths.append(len(path))
        network.augment(path, bottleneck)

    mean_length = sum(edge_lengths) / total_augmenting_paths if total_augmenting_paths else 0
    return max_flow_SAP, total_augmenting_paths, mean_length


def ford_fulkerson_strategies(network: ResidualNetwork) -> Dict[str, Tuple[float, int, float]]:
    """Run all strategies on independent clones of ``network`` for convenience."""

    return {
        "sap": ford_fulkerson(network.clone()),
        "dfs_like": ford_fulkerson_DFS_like(network.clone()),
        "max_capacity": ford_fulkerson_max_capacity(network.clone()),
        "random": ford_fulkerson_random(network.clone()),
    }

