"""Entry point for running Ford-Fulkerson simulations."""

import argparse
import os

from ford_fulkerson.graph_generation import GenerateSinkSourceGraph, visualize_graph
from ford_fulkerson.io import save_graph_data
from ford_fulkerson.runner import run_strategies_for_graph


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


def generate_graphs(visualize: bool = False):
    graph_no = 0
    for n, r, upperCap in SIMULATION_VALUES:
        graph_no += 1
        graph = GenerateSinkSourceGraph(n, r, upperCap)

        if visualize:
            visualize_graph(
                graph.vertices,
                graph.edges,
                graph.capacities,
                graph.source,
                graph.sink,
            )
        save_graph_data(graph, graph_no)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulation datasets and run Ford-Fulkerson strategies."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render generated graphs when datasets are regenerated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    missing_files = False
    for graph_no in range(1, 9):
        for file_type in ["vertices", "edges", "capacities", "adjlist", "meta_info"]:
            file_path = f"sim_val_{graph_no}_{file_type}.csv"
            if not os.path.exists(file_path):
                print("Some or all files missing, regenerating graph and storing the data")
                print(f"file missing : {file_path}")
                missing_files = True
                break

    if missing_files:
        generate_graphs(visualize=args.visualize)

    for graph_no in range(1, 9):
        run_strategies_for_graph(graph_no)


if __name__ == "__main__":
    main()

