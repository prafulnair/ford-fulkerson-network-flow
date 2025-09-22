"""Entry point for running Ford-Fulkerson simulations."""

import os

from ford_fulkerson.algorithms import (
    ford_fulkerson,
    ford_fulkerson_DFS_like,
    ford_fulkerson_max_capacity,
    ford_fulkerson_random,
)
from ford_fulkerson.graph_generation import (
    GenerateSinkSourceGraph,
    breadth_first_search,
    get_sink,
    visualize_graph,
)
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
        V, E, capacities, source, adjlist = GenerateSinkSourceGraph(n, r, upperCap)

        distance, parent = breadth_first_search(V, E, capacities, adjlist, source)
        sink, max_distance = get_sink(distance, parent)
        visualize_graph(V, E, capacities, source, sink)
        max_distance = distance
        total_edges_of_Graph = len(E)
        save_graph_data(
            V,
            E,
            capacities,
            adjlist,
            source,
            sink,
            n,
            r,
            upperCap,
            max_distance,
            total_edges_of_Graph,
            graph_no,
        )


def run_algorithms(graph_no):
    meta_info_path = f"sim_val_{graph_no}_meta_info.csv"
    vertices_path = f"sim_val_{graph_no}_vertices.csv"
    edges_path = f"sim_val_{graph_no}_edges.csv"
    capacities_path = f"sim_val_{graph_no}_capacities.csv"
    adjlist_path = f"sim_val_{graph_no}_adjlist.csv"

    source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
        meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path
    )

    print(f"\nThe source is {source} and the sink is {sink}")

    max_flow_SAP, TAP_SAP, ml_Sap = ford_fulkerson(capacities, source, sink)
    MPL = ml_Sap / maxdist
    print(f"The maximum possible flow for SAP (graph {graph_no}) is {max_flow_SAP}")
    print(f"total augmenting paths for SAP for graph {graph_no} is {TAP_SAP}")
    print(f"Mean Length for SAP for graph {graph_no} is {ml_Sap}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")

    source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
        meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path
    )

    print(f"\nThe source is {source} and the sink is {sink}")

    max_flow_DFS_like, TAP_DFS_like, ml_DFS_like = ford_fulkerson_DFS_like(capacities, source, sink, adjlist)
    MPL = ml_DFS_like / maxdist
    print(f"The maximum possible flow for DFS (graph {graph_no}) is {max_flow_DFS_like}")
    print(f"total augmenting paths for DFS-like for graph {graph_no} is {TAP_DFS_like}")
    print(f"Mean Length for DFS-like graph {graph_no} is {ml_DFS_like}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")

    source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
        meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path
    )

    print(f"\nThe source is {source} and the sink is {sink}")
    max_flow_maxcap, TAP_maxCap, ml_maxCap = ford_fulkerson_max_capacity(capacities, source, sink, adjlist)
    MPL = ml_maxCap / maxdist
    print(f"The maximum possible flow for maxCap (graph {graph_no}) is {max_flow_maxcap}")
    print(f"total augmenting paths for maxCap for graph {graph_no} is {TAP_maxCap}")
    print(f"Mean Length for maxCap graph {graph_no} is {ml_maxCap}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")

    source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
        meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path
    )

    print(f"\nThe source is {source} and the sink is {sink}")

    max_flow_random, TAP_random, ml_random = ford_fulkerson_random(capacities, source, sink, adjlist)
    MPL = ml_random / maxdist
    print(f"The maximum possible flow for random (graph {graph_no}) is {max_flow_random}")
    print(f"total augmenting paths for random for graph {graph_no} is {TAP_random}")
    print(f"Mean Length for random graph {graph_no} is {ml_random}")
    print(f"MPL for graph {graph_no} is {MPL}")
    print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")


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

