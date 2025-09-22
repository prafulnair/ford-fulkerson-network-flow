"""Helpers for persisting and loading graph data to CSV files."""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Tuple

from .models import GraphInstance

Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]


def save_graph_data(graph: GraphInstance, graph_no: int) -> None:
    """Persist ``graph`` and its metadata to CSV files for later reuse."""

    meta_distance = graph.max_distance
    if meta_distance is None or math.isinf(meta_distance):
        meta_distance_value = "Inf"
    else:
        meta_distance_value = meta_distance

    with open(f"sim_val_{graph_no}_meta_info.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "source_x_coordinate",
                "source_y_coordinate",
                "sink_x_coordinate",
                "sink_y_coordinate",
                "n",
                "r",
                "uppercap",
                "longest_pathToSink",
                "total_edges",
            ]
        )
        writer.writerow(
            [
                graph.source[0],
                graph.source[1],
                graph.sink[0],
                graph.sink[1],
                graph.n,
                graph.r,
                graph.upper_cap,
                meta_distance_value,
                graph.total_edges,
            ]
        )
        print("meta info saved")

    with open(f"sim_val_{graph_no}_vertices.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x-coordinate", "y-coordinate"])
        for vertex in graph.vertices:
            writer.writerow(vertex)

    with open(f"sim_val_{graph_no}_edges.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["v1_x", "v1_y", "v2_x", "v2_y"])
        for edge in graph.edges:
            writer.writerow([edge[0][0], edge[0][1], edge[1][0], edge[1][1]])

    with open(f"sim_val_{graph_no}_capacities.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["v1_x", "v1_y", "v2_x", "v2_y", "capacity"])
        for edge, capacity in graph.capacities.items():
            writer.writerow([edge[0][0], edge[0][1], edge[1][0], edge[1][1], capacity])

    with open(f"sim_val_{graph_no}_adjlist.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["vertex_x", "vertex_y", "pairs of vertices coordinates"])
        for vertex, neighbors in graph.adjacency_list.items():
            flattened: List[float] = []
            for neighbour in neighbors:
                flattened.extend([neighbour[0], neighbour[1]])
            writer.writerow([vertex[0], vertex[1], *flattened])
    print("Info saved")


def read_data(
    file_path1: str,
    file_path2: str,
    file_path3: str,
    file_path4: str,
    file_path5: str,
) -> GraphInstance:
    """Reconstruct a :class:`GraphInstance` from CSV files produced by :func:`save_graph_data`."""

    with open(file_path1, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        row = next(reader)
        source: Vertex = (float(row[0]), float(row[1]))
        sink: Vertex = (float(row[2]), float(row[3]))
        n = int(float(row[4]))
        r = float(row[5])
        upperCap = float(row[6])
        maxdistance_raw = row[7]
        try:
            maxdist = float("inf") if maxdistance_raw == "Inf" else float(maxdistance_raw)
        except ValueError:
            maxdist = 18.0
        try:
            total_edges = int(float(row[8]))
        except ValueError:
            total_edges = 0

    vertices: List[Vertex] = []
    with open(file_path2, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            x, y = map(float, row)
            vertices.append((x, y))

    edges: List[Edge] = []
    with open(file_path3, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            edges.append((u, v))

    capacities: Dict[Edge, int] = {}
    with open(file_path4, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            capacity = int(row[4])
            capacities[(u, v)] = capacity

    adjacency_list: Dict[Vertex, List[Vertex]] = {}
    with open(file_path5, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            vertex = (float(row[0]), float(row[1]))
            adjacent_vertices = [
                (float(row[i]), float(row[i + 1])) for i in range(2, len(row), 2)
            ]
            adjacency_list[vertex] = adjacent_vertices

    return GraphInstance(
        vertices=vertices,
        edges=edges,
        capacities=capacities,
        adjacency_list=adjacency_list,
        source=source,
        sink=sink,
        n=n,
        r=r,
        upper_cap=upperCap,
        max_distance=maxdist,
        total_edges=total_edges or len(edges),
    )

