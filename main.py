"""Entry point for running Ford-Fulkerson simulations."""

import os

from math import isinf
from typing import Optional

from ford_fulkerson.algorithms import (
    ford_fulkerson,
    ford_fulkerson_DFS_like,
    ford_fulkerson_max_capacity,
    ford_fulkerson_random,
)
from ford_fulkerson.graph_generation import GenerateSinkSourceGraph, visualize_graph
from ford_fulkerson.io import read_data, save_graph_data


SIMULATION_VALUES = [
    [100, 0.2, 2],
    [200, 0.2, 2],
    [100, 0.3, 2],
    [200, 0.3, 2],
    [100, 0.2, 50],
    [200, 0.2, 50],
    [100, 0.3, 50],
    [200, 0.3, 50],
]


def generate_graphs():
    graph_no = 0
    for n, r, upperCap in SIMULATION_VALUES:
        graph_no += 1
        graph = GenerateSinkSourceGraph(n, r, upperCap)

        visualize_graph(graph.vertices, graph.edges, graph.capacities, graph.source, graph.sink)
        save_graph_data(graph, graph_no)


def run_algorithms(graph_no):
    meta_info_path = f"sim_val_{graph_no}_meta_info.csv"
    vertices_path = f"sim_val_{graph_no}_vertices.csv"
    edges_path = f"sim_val_{graph_no}_edges.csv"
    capacities_path = f"sim_val_{graph_no}_capacities.csv"
    adjlist_path = f"sim_val_{graph_no}_adjlist.csv"

    graph = read_data(meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path)

    print(f"\nThe source is {graph.source} and the sink is {graph.sink}")

    residual = graph.create_residual_network()

    max_flow_SAP, TAP_SAP, ml_Sap = ford_fulkerson(residual.clone())
    MPL = _compute_mpl(graph.max_distance, ml_Sap)
    print(f"The maximum possible flow for SAP (graph {graph_no}) is {max_flow_SAP}")
    print(f"total augmenting paths for SAP for graph {graph_no} is {TAP_SAP}")
    print(f"Mean Length for SAP for graph {graph_no} is {ml_Sap}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {graph.total_edges}")

    max_flow_DFS_like, TAP_DFS_like, ml_DFS_like = ford_fulkerson_DFS_like(residual.clone())
    MPL = _compute_mpl(graph.max_distance, ml_DFS_like)
    print(f"\nThe source is {graph.source} and the sink is {graph.sink}")
    print(f"The maximum possible flow for DFS (graph {graph_no}) is {max_flow_DFS_like}")
    print(f"total augmenting paths for DFS-like for graph {graph_no} is {TAP_DFS_like}")
    print(f"Mean Length for DFS-like graph {graph_no} is {ml_DFS_like}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {graph.total_edges}")

    max_flow_maxcap, TAP_maxCap, ml_maxCap = ford_fulkerson_max_capacity(residual.clone())
    MPL = _compute_mpl(graph.max_distance, ml_maxCap)
    print(f"\nThe source is {graph.source} and the sink is {graph.sink}")
    print(f"The maximum possible flow for maxCap (graph {graph_no}) is {max_flow_maxcap}")
    print(f"total augmenting paths for maxCap for graph {graph_no} is {TAP_maxCap}")
    print(f"Mean Length for maxCap graph {graph_no} is {ml_maxCap}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {graph.total_edges}")

    max_flow_random, TAP_random, ml_random = ford_fulkerson_random(residual.clone())
    MPL = _compute_mpl(graph.max_distance, ml_random)
    print(f"\nThe source is {graph.source} and the sink is {graph.sink}")
    print(f"The maximum possible flow for random (graph {graph_no}) is {max_flow_random}")
    print(f"total augmenting paths for random for graph {graph_no} is {TAP_random}")
    print(f"Mean Length for random graph {graph_no} is {ml_random}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {graph.total_edges}")


def _compute_mpl(max_distance: Optional[float], mean_length: float) -> float:
    if max_distance in (None, 0):
        return 0
    if isinf(float(max_distance)):
        return 0
    return mean_length / float(max_distance)


def main():
    graph_exist = True
    for graph_no in range(1, 9):
        for file_type in ["vertices", "edges", "capacities", "adjlist", "meta_info"]:
            file_path = f"sim_val_{graph_no}_{file_type}.csv"
            if not os.path.exists(file_path):
                print("Some or all files missing, regenerating graph and storing the data")
                print(f"file missing : {file_path}")
                graph_exist = False
                break

    if not graph_exist:
        generate_graphs()

    if graph_exist:
        for graph_no in range(1, 9):
            run_algorithms(graph_no)


if __name__ == "__main__":
    main()

