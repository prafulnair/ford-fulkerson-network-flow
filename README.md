# Ford-Fulkerson Algorithm with Random Graph Generation and Diverse Augmenting Path Strategies

This repository contains an implementation of the Ford-Fulkerson Algorithm for network flow optimization, along with a random graph generator. The project integrates four augmenting path strategies: SAP (Shortest Augmenting Path), DFS-like, MaxCap (Maximum Capacity), and Random. Comprehensive testing and evaluation are conducted across multiple simulation sets.

## Project Overview

### Features
- **Algorithm Implementation**: Ford-Fulkerson Algorithm with four augmenting path strategies.
  - **SAP (Shortest Augmenting Path)**
  - **DFS-like**
  - **MaxCap (Maximum Capacity)**
  - **Random**
- **Random Graph Generation**: Generates random source-sink graphs with node counts ranging from 200 to 500, using (x, y) coordinates and random capacities.
- **Testing and Evaluation**: Analyzes algorithm correctness and performance across eight simulation sets and a customized set.

### Files
- `ford_fulkerson/`: Python package that hosts the core modules.
  - `algorithms.py`: Augmenting-path strategies (SAP, DFS-like, Random, MaxCap).
  - `graph_generation.py`: Random graph generation helpers and optional visualization utilities.
  - `io.py`: CSV helpers for persisting and loading graph data.
  - `models.py`: Dataclasses representing immutable graphs and mutable residual networks.
- `main.py`: Command line entry point that wires the package modules together.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python libraries.

### Dependencies
- `matplotlib`: For visualization purposes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ford-fulkerson-network-flow.git
   cd ford-fulkerson-network-flow
2. Installed the required libraries
3.       pip install -r requirements.txt

## Usage

Run the simulations through the entry point:

```bash
python main.py
```

The script checks for pre-generated simulation data (stored as CSV files) and, if absent, will create new graphs using the package helpers before executing every augmenting-path strategy.

## Functions Overview
#### Augmenting Path Strategies (`ford_fulkerson/algorithms.py`)

All strategies now operate on a `ResidualNetwork` created from a `GraphInstance`.

    DFS-like: ford_fulkerson_DFS_like(residual_network)
    Random: ford_fulkerson_random(residual_network)
    MaxCap: ford_fulkerson_max_capacity(residual_network)
    SAP: ford_fulkerson(residual_network)

#### Graph Generation (`ford_fulkerson/graph_generation.py`)

    GenerateSinkSourceGraph(n, r, upperCap): Generates a random source-sink graph as a GraphInstance.
    breadth_first_search(V, E, capacities, adjlist, source): Helper used to determine sink candidates.
    get_sink(distance, parent, source): Returns the sink at the end of the longest acyclic path from the source.

#### Visualization (Optional)

    visualize_graph(vertices, edges, capacities, source, sink): Visualizes the generated graph (for debugging and presentation purposes).

#### Data Persistence (`ford_fulkerson/io.py`)

    save_graph_data(graph, graph_no): Persist generated graph data to CSV files.
    read_data(...): Read graph data from the CSV files created by `save_graph_data` and return a GraphInstance.
