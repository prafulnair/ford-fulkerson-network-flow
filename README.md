# Ford-Fulkerson Algorithm with Random Graph Generation and Diverse Augmenting Path Strategies

This repository contains an implementation of the Ford-Fulkerson algorithm for network flow optimisation together with tools to
generate random source-sink graphs and benchmark multiple augmenting-path strategies. The codebase now ships as a small Python
package so the graph models, IO helpers, and algorithm variants can be reused from scripts or notebooks in addition to the
provided command-line runner.

## Project Overview

### Module architecture
The project is organised as the `ford_fulkerson` package plus a lightweight entry point:

- `ford_fulkerson.algorithms` implements the four augmenting-path strategies (SAP, DFS-like, max capacity, and random). Each
  function consumes a `ResidualNetwork` from `ford_fulkerson.models` and returns flow and path metrics.
- `ford_fulkerson.graph_generation` is responsible for synthesising random graphs, measuring source/sink candidates, and
  (optionally) visualising the result using Matplotlib.
- `ford_fulkerson.models` defines immutable `GraphInstance` snapshots and mutable `ResidualNetwork` objects that all other
  modules share when reading, writing, or augmenting flows.
- `ford_fulkerson.io` persists generated graphs to CSV and reloads them as `GraphInstance` objects so simulations can be
  reproduced.
- `ford_fulkerson.runner` loads stored graphs once, clones residual networks, and executes every enabled strategy while
  collecting comparable metrics.
- `main.py` checks that all eight simulation datasets exist, triggers graph generation if they do not, and delegates execution
  to `ford_fulkerson.runner`.

### Simulation scenarios and data
Eight simulation sets are bundled under `graph_data/`. They correspond to every combination of:

| Nodes (`n`) | Radius (`r`) | Capacity upper bound |
|-------------|--------------|----------------------|
| 100         | 0.2          | 2                    |
| 100         | 0.2          | 50                   |
| 100         | 0.3          | 2                    |
| 100         | 0.3          | 50                   |
| 200         | 0.2          | 2                    |
| 200         | 0.2          | 50                   |
| 200         | 0.3          | 2                    |
| 200         | 0.3          | 50                   |

The runner loads these CSV files when present; otherwise, `main.py` regenerates them using the same parameter grid. Each file set
contains vertices, edges, capacities, adjacency lists, and metadata (such as maximum source-to-sink distance) for reproducible
experiments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ford-fulkerson-network-flow.git
   cd ford-fulkerson-network-flow
   ```
2. (Optional) Create and activate a virtual environment.
3. Install the runtime dependency (required for graph visualisation invoked by `main.py`):
   ```bash
   pip install matplotlib
   ```

## Usage

Run the bundled simulations through the entry point:

```bash
python main.py
```

The script will reuse the CSV data in `graph_data/` when available. If any file is missing it regenerates the corresponding
graphs, saves them through `ford_fulkerson.io`, and then executes every augmenting-path strategy via `ford_fulkerson.runner`.
Dataset regeneration runs headlessly by default; pass `--visualize` to open Matplotlib windows for each new graph:

```bash
python main.py --visualize
```

## Strategy and utility reference

- **Augmenting Path Strategies (`ford_fulkerson.algorithms`)**
  - `ford_fulkerson`: Shortest augmenting path (SAP) using BFS.
  - `ford_fulkerson_DFS_like`: Depth-first bias using a Dijkstra-style traversal.
  - `ford_fulkerson_max_capacity`: Greedy augmentation that always chooses the remaining path with maximum bottleneck capacity.
  - `ford_fulkerson_random`: Randomised tie-breaking on augmenting paths.
- **Graph Generation (`ford_fulkerson.graph_generation`)**
  - `GenerateSinkSourceGraph(n, r, upperCap)`: Creates a random graph and returns it as a `GraphInstance`.
  - `visualize_graph(...)`: Optional Matplotlib helper for inspecting generated graphs.
- **Data Persistence (`ford_fulkerson.io`)**
  - `save_graph_data(graph, graph_no)`: Serialises vertices, edges, capacities, adjacency lists, and metadata to CSV.
  - `read_data(...)`: Loads the CSV bundle for a graph ID and reconstructs a `GraphInstance`.
- **Simulation Runner (`ford_fulkerson.runner`)**
  - `run_strategies_for_graph(graph_no, strategies=None)`: Loads the stored graph once, clones the residual network per
    strategy, and returns comparable metrics (max flow, augmenting paths, mean path length, MPL).

## Recent changes

- The project previously lived as a single monolithic script. It has been reorganised into the modules described above so that
  algorithms, models, graph generation, and IO code can be reused independently and tested in isolation.

