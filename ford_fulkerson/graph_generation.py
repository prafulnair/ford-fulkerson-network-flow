"""Utilities for generating and inspecting random source-sink graphs."""

import random
from collections import deque

import matplotlib.pyplot as plt


def visualize_graph(V, E, capacities, source, sink):
    """Render the generated graph using matplotlib (for debugging/visualization)."""
    for vertex in V:
        plt.scatter(*vertex, color="blue", marker="o")

    for edge in E:
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


def GenerateSinkSourceGraph(n, r, upperCap):
    """Generate a random directed graph with source and sink candidates."""
    Vertices = []
    E = set()
    capacities = {}
    adjacency_list = {}

    for _ in range(n):
        node = (round(random.uniform(0, 1), 4), round(random.uniform(0, 1), 4))
        Vertices.append(node)
    for v in Vertices:
        adjacency_list[v] = []

    for u in Vertices:
        for v in Vertices:
            if u != v and (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2 <= r ** 2:
                rand = random.uniform(0, 1)
                if rand < 0.5:
                    edge1 = (u, v)
                    edge2 = (v, u)
                    if (edge1 not in E) and (edge2 not in E):
                        E.add(edge1)
                        if u in adjacency_list:
                            adjacency_list[u].append(v)
                        else:
                            adjacency_list[u] = [v]
                else:
                    edge1 = (v, u)
                    edge2 = (u, v)
                    if (edge1 not in E) and (edge2 not in E):
                        E.add(edge1)
                        if v in adjacency_list:
                            adjacency_list[v].append(u)
                        else:
                            adjacency_list[v] = [u]

    for edge in E:
        capacities[edge] = int(round(random.uniform(1, upperCap), 4))

    source = random.choice(Vertices)

    return Vertices, E, capacities, source, adjacency_list


def get_sink(distance, parent):
    """Return the sink at the end of the longest acyclic path from the source."""
    max_distance = max(distance.values())

    sink = [node for node, value in distance.items() if value == max_distance][0]

    return sink, max_distance


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

