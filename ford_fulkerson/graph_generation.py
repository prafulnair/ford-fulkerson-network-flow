"""Utilities for generating and inspecting random source-sink graphs."""

import random
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple
from typing import Optional


import matplotlib.pyplot as plt

from .models import GraphInstance

Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]


def visualize_graph(
    vertices: Sequence[Vertex],
    edges: Sequence[Edge],
    capacities,
    source: Vertex,
    sink: Vertex,
):
    """Render the generated graph using matplotlib (for debugging/visualization)."""
    for vertex in vertices:
        plt.scatter(*vertex, color="blue", marker="o")

    for edge in edges:
        plt.arrow(
            *edge[0],
            edge[1][0] - edge[0][0],
            edge[1][1] - edge[0][1],
            head_width=0.015,
            head_length=0.015,
            fc="red",
            ec="red",
        )

    plt.scatter(*source, color="yellow", marker="s", label="Source")
    plt.scatter(*sink, color="purple", marker="s", label="Sink")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Randomly Generated Source-Sink Graph")
    plt.legend()
    plt.show()


def GenerateSinkSourceGraph(
        
    n: int, r: float, upperCap: float, seed: Optional[int] = None
) -> GraphInstance:
    """Generate a random directed graph with source and sink candidates."""

    if seed is not None:
        random.seed(seed)

    vertices: List[Vertex] = []
    edges: Set[Edge] = set()
    capacities: Dict[Edge, int] = {}
    adjacency_list: Dict[Vertex, List[Vertex]] = {}

    for _ in range(n):
        node = (round(random.uniform(0, 1), 4), round(random.uniform(0, 1), 4))
        vertices.append(node)
        adjacency_list[node] = []

    for u in vertices:
        for v in vertices:
            if u == v:
                continue
            if (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2 <= r**2:
                rand = random.uniform(0, 1)
                edge = (u, v) if rand < 0.5 else (v, u)
                reverse = (edge[1], edge[0])
                if edge not in edges and reverse not in edges:
                    edges.add(edge)
                    adjacency_list.setdefault(edge[0], []).append(edge[1])

    for edge in edges:
        capacities[edge] = int(round(random.uniform(1, upperCap), 4))

    source = random.choice(vertices)
    distance, parent = breadth_first_search(vertices, list(edges), capacities, adjacency_list, source)
    sink, max_distance = get_sink(distance, parent, source)

    return GraphInstance(
        vertices=vertices,
        edges=list(edges),
        capacities=capacities,
        adjacency_list=adjacency_list,
        source=source,
        sink=sink,
        n=n,
        r=r,
        upper_cap=upperCap,
        max_distance=max_distance,
        total_edges=len(edges),
        seed=seed,
    )


def get_sink(distance, parent, source):
    """Return the sink at the end of the longest acyclic path from the source."""

    finite_distances = {node: dist for node, dist in distance.items() if dist != float("inf")}
    if not finite_distances:
        return source, 0

    max_distance = max(finite_distances.values(), default=0)
    for node, value in finite_distances.items():
        if value == max_distance:
            return node, max_distance

    return source, 0


def breadth_first_search(V, E, capacities, adjlist, source):
    """Breadth-first search used to compute distances and parents from the source."""
    visited = set()
    queue = deque([source])

    distance = {node: float("inf") for node in V}
    parent = {node: None for node in V}

    visited.add(source)
    queue.append(source)
    distance[source] = 0
    parent[source] = None

    while queue:
        node = queue.popleft()

        for neighbours in adjlist[node]:
            if neighbours not in visited:
                visited.add(neighbours)
                queue.append(neighbours)

                distance[neighbours] = distance[node] + 1
                parent[neighbours] = node

    return distance, parent

