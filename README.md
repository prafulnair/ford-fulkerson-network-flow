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
- `main.py`: Main implementation file containing the Ford-Fulkerson Algorithm and graph generation code.
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

## Functions Overview
#### Augmenting Path Strategies

    DFS-like: ford_fulkerson_DFS_like(capacities, source, sink, adjlist)
    Random: ford_fulkerson_random(capacities, source, sink, adjlist)
    MaxCap: ford_fulkerson_max_capacity(capacities, source, sink, adjList)
    SAP: ford_fulkerson(capacities, source, sink)

#### Graph Generation

    GenerateSinkSourceGraph(n, r, upperCap): Generates a random source-sink graph.

#### Visualization (Optional)

    visualize_graph(V, E, capacities, source, sink): Visualizes the generated graph (for debugging and presentation purposes).
