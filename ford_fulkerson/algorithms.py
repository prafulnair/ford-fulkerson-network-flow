"""Augmenting path strategies for the Ford-Fulkerson algorithm."""

from collections import defaultdict
import heapq
import random


def dijkstra_foundation_DFSLike(source, sink, adjlist, capacities, parent, typee):
    """Shared Dijkstra-style search used by DFS-like and random strategies."""
    distances = {vex: float("Inf") for vex in adjlist}
    distances[source] = 0
    max_capacity = max(capacities.values())
    counter = 0
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        for v in adjlist[u]:
            residual_capacity = capacities.get((u, v), 0) - capacities.get((v, u), 0)
            if distances[v] == float("Inf"):
                if residual_capacity > 0 and distances[u] + 1 < distances[v]:
                    distances[v] = distances[u] + 1
                    parent[v] = u
                    if typee == "dfs_like":
                        distances[v] = distances[v] - counter
                        counter -= 1
                    else:
                        distances[v] = distances[v] - random.randint(0, max_capacity)
                        counter -= 1
                    heapq.heappush(priority_queue, (distances[v], v))

    path = []
    current = sink

    while True:
        path.insert(0, current)
        current = parent[current]

        if current is None:
            break
    if distances[sink] != float("Inf"):
        return True
    return False


def ford_fulkerson_random(capacities, source, sink, adjlist):
    """Ford-Fulkerson using randomized tie-breaking for augmenting paths."""
    graph = capacities
    random.choice(list(capacities.values()))
    parent = defaultdict(lambda: None)
    max_flow_random = 0
    typee = "random"

    edge_lengths = []

    total_augmenting_paths = 0
    edges = 0
    while dijkstra_foundation_DFSLike(source, sink, adjlist, capacities, parent, typee):
        edges = 0
        total_augmenting_paths += 1
        counter_var = max(capacities.values())
        flow_path = float("Inf")
        t = sink
        while t != source:
            edges += 1
            flow_path = min(flow_path, graph[parent[t], t])
            t = parent[t]
        max_flow_random += flow_path
        edge_lengths.append(edges)

        vex = sink
        while vex != source:
            u = parent[vex]
            graph[u, vex] -= flow_path

            if (vex, u) not in graph:
                graph[vex, u] = flow_path
            else:
                graph[vex, u] += flow_path
            vex = parent[vex]

    total_edges = 0
    for edge in edge_lengths:
        total_edges += edge
    if total_augmenting_paths == 0:
        print("Total augmenting path for random coming as zero")
    mean_length = total_edges / total_augmenting_paths if total_augmenting_paths > 0 else 0

    return max_flow_random, total_augmenting_paths, mean_length


def get_residual_capacity(E, u, v):
    return E.get((u, v), 0)


def dijkstra_maxCap(capacities, adjList, source, sink):
    priority_queue = []
    initial_element = (-float("inf"), source, [])
    priority_queue.append(initial_element)

    heapq.heapify(priority_queue)

    capacity = {}
    for v in adjList.keys():
        capacity[v] = 0

    capacity[source] = float("Inf")

    while priority_queue:
        curr_capacity, curr_vertex, curr_path = heapq.heappop(priority_queue)

        if curr_vertex == sink:
            return capacity[sink], curr_path

        for neighbor in adjList.get(curr_vertex, []):
            edge_capacity = get_residual_capacity(capacities, curr_vertex, neighbor)
            min_capacity = min(capacity[curr_vertex], edge_capacity)
            neighborval = tuple(neighbor)
            if min_capacity > capacity.get(neighborval, 0):
                capacity[neighborval] = min_capacity
                new_path = curr_path + [(curr_vertex, neighborval)]
                heapq.heappush(priority_queue, (-min_capacity, neighborval, new_path))

    return 0, []


def ford_fulkerson_max_capacity(capacities, source, sink, adjList):
    graph = capacities

    total_augmenting_paths = 0
    total_length = 0

    max_flow_max_capacity = 0

    while True:
        capacity, augmenting_path = dijkstra_maxCap(graph, adjList, source, sink)

        if capacity == 0:
            break

        total_length += len(augmenting_path)
        total_augmenting_paths += 1
        max_flow_max_capacity += capacity

        for u, v in augmenting_path:
            residual_capacity = get_residual_capacity(graph, u, v)
            if (u, v) in graph:
                graph[u, v] -= capacity
            if (v, u) in graph:
                graph[v, u] += capacity

    mean_length = total_length / total_augmenting_paths if total_augmenting_paths > 0 else 0
    return max_flow_max_capacity, total_augmenting_paths, mean_length


def ford_fulkerson_DFS_like(capacities, source, sink, adjlist):
    graph = capacities
    max(capacities.values())

    edge_length = []
    edges = 0
    total_augmenting_paths = 0

    parent = defaultdict(lambda: None)
    max_flow_DFS_like = 0
    typee = "dfs_like"
    while dijkstra_foundation_DFSLike(source, sink, adjlist, capacities, parent, typee):
        edges = 0
        total_augmenting_paths += 1
        counter_var = max(capacities.values())
        flow_path = float("Inf")
        t = sink
        while t != source:
            edges += 1
            flow_path = min(flow_path, graph[parent[t], t])
            t = parent[t]
        max_flow_DFS_like += flow_path
        edge_length.append(edges)

        vex = sink
        while vex != source:
            u = parent[vex]
            graph[u, vex] -= flow_path

            if (vex, u) not in graph:
                graph[vex, u] = flow_path
            else:
                graph[vex, u] += flow_path
            vex = parent[vex]

    total_edges = 0
    for edges in edge_length:
        total_edges += edges
    mean_length = total_edges / total_augmenting_paths if total_augmenting_paths > 0 else 0
    return max_flow_DFS_like, total_augmenting_paths, mean_length


def BFS_FF_SAP(source, sink, parent, graph, vertices):
    visited = set()
    queue = []

    queue.append(source)
    visited.add(source)

    while queue:
        u = queue.pop(0)

        for v in vertices:
            if v not in visited and graph.get((u, v), 0) > 0:
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False


def ford_fulkerson(capacities, source, sink):
    total_augmenting_paths = 0
    graph = capacities
    vertices = set(v for edge in graph.keys() for v in edge)
    parent = defaultdict(lambda: None)
    max_flow_SAP = 0
    edge_lengths = []
    edges = 0
    while BFS_FF_SAP(source, sink, parent, graph, vertices):
        total_augmenting_paths = total_augmenting_paths + 1
        flow_path = float("Inf")
        t = sink
        while t != source:
            edges = edges + 1
            flow_path = min(flow_path, graph[parent[t], t])
            t = parent[t]
        max_flow_SAP += flow_path
        edge_lengths.append(edges)

        vex = sink
        while vex != source:
            u = parent[vex]
            graph[u, vex] -= flow_path

            if (vex, u) not in graph:
                graph[vex, u] = flow_path
            else:
                graph[vex, u] += flow_path
            vex = parent[vex]
    total_edges = 0
    for edge in edge_lengths:
        total_edges = total_edges + edges
    mean_length = total_edges / total_augmenting_paths if total_augmenting_paths > 0 else 0
    return max_flow_SAP, total_augmenting_paths, mean_length

