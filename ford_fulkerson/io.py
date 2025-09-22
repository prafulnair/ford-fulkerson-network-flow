"""Helpers for persisting and loading graph data to CSV files."""

import csv


def save_graph_data(
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
):
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
                source[0],
                source[1],
                sink[0],
                sink[1],
                n,
                r,
                upperCap,
                max_distance,
                total_edges_of_Graph,
            ]
        )
        print("meta info saved")

    with open(f"sim_val_{graph_no}_vertices.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x-coordinate", "y-coordinate"])
        for v in V:
            writer.writerow(v)

    with open(f"sim_val_{graph_no}_edges.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["v1_x", "v1_y", "v2_x", "v2_y"])
        for edge in E:
            writer.writerow([edge[0][0], edge[0][1], edge[1][0], edge[1][1]])

    with open(f"sim_val_{graph_no}_capacities.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["v1_x", "v1_y", "v2_x", "v2_y", "capacity"])
        for edge, capacity in capacities.items():
            writer.writerow([edge[0][0], edge[0][1], edge[1][0], edge[1][1], capacity])

    with open(f"sim_val_{graph_no}_adjlist.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["vertex_x", "vertex_y", "pairs of vertices coordinates"])
        for vertex, neighbors in adjlist.items():
            writer.writerow([vertex[0], vertex[1], *[neighbor for n in neighbors for neighbor in n]])
    print("Info saved")


def read_data(file_path1, file_path2, file_path3, file_path4, file_path5):
    with open(file_path1, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            source = (float(row[0]), float(row[1]))
            sink = (float(row[2])), float(row[3])
            n = float(row[4])
            r = float(row[5])
            upperCap = float(row[6])
            maxdistance = row[7]
            if maxdistance == "Inf":
                maxdist = 18
            else:
                try:
                    maxdist = float(maxdistance)
                except ValueError:
                    maxdist = 18
            total_edges = float(row[8])

    V = []
    with open(file_path2, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            x, y = map(float, row)
            V.append((x, y))

    E = set()
    with open(file_path3, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            E.add((u, v))

    capacities = {}
    with open(file_path4, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            capacity = int(row[4])
            capacities[(u, v)] = capacity

    adjlist = {}
    with open(file_path5, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            vertex = (float(row[0]), float(row[1]))
            adjacent_vertices = [(float(row[i]), float(row[i + 1])) for i in range(2, len(row), 2)]
            adjlist[vertex] = adjacent_vertices

    return source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges

