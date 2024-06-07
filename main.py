
import math
import random
import csv
import os
# Matplotlib is the only external library being used but that is only for visualization purpose. It Adds nothing to the solving of the problem
import matplotlib.pyplot as plt 
from collections import defaultdict, deque
import os
import heapq

""" The following method is used to find augmenting path for DFS-like and Random.
    based on which method its calling and what's the type set to, it calculates accordingly and returns results./
"""
def dijkstra_foundation_DFSLike(source, sink, adjlist, capacities, parent, typee):
        
    distances = {vex: float("Inf") for vex in adjlist}
    distances[source] = 0
    max_capacity = max(capacities.values())
    counter = 0
    PQ = [(0, source)]

    while PQ:
        current_distance, u = heapq.heappop(PQ)

        for v in adjlist[u]:
            residual_capacity = capacities.get((u, v), 0) - capacities.get((v, u), 0)
            if distances[v] == float("Inf"):
                if residual_capacity > 0 and distances[u] + 1 < distances[v]:
                    distances[v] = distances[u] + 1
                    parent[v] = u
                    if typee == 'dfs_like':
                        distances[v] = distances[v] - counter
                        counter -= 1
                    else:
                        distances[v] = distances[v] - random.randint(0,max_capacity)
                        counter-=1
                    heapq.heappush(PQ, (distances[v], v))

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
    graph = capacities
    vertices = set(v for edge in graph.keys() for v in edge)
    counter_var = random.choice(list(capacities.values()))
    parent = defaultdict(lambda: None)
    max_flow_random = 0
    typee = 'random'

    edge_lengths = []
    
    total_augmenting_Paths = 0
    edges = 0
    #while DFS_random(source, sink, parent, graph, adjlist, counter_var, capacities):
    while dijkstra_foundation_DFSLike(source, sink, adjlist,capacities,parent,typee):
        #print("entered while loop of random")
        edges = 0
        total_augmenting_Paths += 1
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
    if total_augmenting_Paths == 0:
        print("Total augmenting path for random coming as zero")
    mean_length = total_edges/total_augmenting_Paths if total_augmenting_Paths > 0 else 0

    return max_flow_random, total_augmenting_Paths, mean_length

###############################################
"""The following set of 3 methods work to solve maxCap algorithm"""
def get_residual_capacity(E, u, v):
    return E.get((u, v), 0)

def dijkstra_maxCap(capacities, adjList, source, sink):

    PQ = []
    initial_element = (-float("inf"), source, [])
    PQ.append(initial_element)

    heapq.heapify(PQ)

    capacity = {}
    for v in adjList.keys():
        capacity[v] = 0

    capacity[source] = float("Inf")

    while PQ:
        curr_capacity, curr_vertex, curr_path = heapq.heappop(PQ)

        if curr_vertex == sink:
            return capacity[sink], curr_path

        for neighbor in adjList.get(curr_vertex, []):
            edge_capacity = get_residual_capacity(capacities, curr_vertex, neighbor)
            min_capacity = min(capacity[curr_vertex], edge_capacity)
            neighborval = tuple(neighbor)
            if min_capacity > capacity.get(neighborval, 0):
                capacity[neighborval] = min_capacity
                new_path = curr_path + [(curr_vertex, neighborval)]
                # possible bug is solved here. 
                heapq.heappush(PQ, (-min_capacity, neighborval, new_path))

    return 0, []

def ford_fulkerson_max_capacity(capacities, source, sink, adjList):
    graph = capacities
    vertices = set(v for edge in graph.keys() for v in edge)

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
    
    mean_length = total_length/total_augmenting_paths if total_augmenting_paths > 0 else 0
    return max_flow_max_capacity, total_augmenting_paths, mean_length


##########################################################################


####################################

"""The following method is for DFS-like. It make use of dijkstra_foundation_DFSLike, just like ford_fulkerson_random"""

def ford_fulkerson_DFS_like(capacities, source, sink, adjlist):
    graph = capacities
    vertices = set(v for edge in graph.keys() for v in edge)
    counter_var = max(capacities.values())

    edge_length = []
    edges = 0
    total_augmenting_paths = 0

    parent = defaultdict(lambda: None)
    max_flow_DFS_like = 0
    typee = 'dfs_like'
    # while DFS_like(source, sink, parent, graph, adjlist, counter_var, capacities):
    while dijkstra_foundation_DFSLike(source, sink, adjlist,capacities,parent,typee):
        edges = 0
        total_augmenting_paths += 1
        counter_var = max(capacities.values())
        flow_path = float("Inf")
        t = sink
        while t != source:
            edges+=1
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
    mean_length = total_edges/ total_augmenting_paths if total_augmenting_paths > 0 else 0
    return max_flow_DFS_like, total_augmenting_paths, mean_length

"""This is the helper method for Ford_fulkerson for SAP"""
def BFS_FF_SAP (source, sink, parent, graph,vertices):
        visited = set()
        queue = []

        queue.append(source)
        visited.add(source)

        while queue:
            u = queue.pop(0)

            for v in vertices:
                if v not in visited and graph.get((u, v),0) > 0:
                    queue.append(v)
                    #add v to visited set
                    visited.add(v) 
                    parent[v] = u
                    if v == sink:
                        # there exists a path
                        #print("returning true")
                        return True
        #print("returning false")
        return False 

""" This is Ford Fulkerson for SAP. This make use of BFS_FF_SAP"""
def ford_fulkerson(capacities, source, sink):
    
    total_augmenting_Paths = 0
    # initializing graph for ford_fulkerson
    graph = capacities
    vertices = set(v for edge in graph.keys() for v in edge)
    parent = defaultdict(lambda: None)
    #initialize flow = 0
    max_flow_SAP = 0
    edge_lengths  = []
    edges = 0
    while BFS_FF_SAP(source, sink, parent, graph, vertices):
        total_augmenting_Paths = total_augmenting_Paths + 1
        flow_path = float("Inf")
        t = sink
        while t != source:
            edges = edges + 1
            flow_path = min(flow_path, graph[parent[t], t])
            t = parent[t]
        max_flow_SAP +=flow_path
        edge_lengths.append(edges)

        vex = sink
        while vex != source:
            u = parent[vex]
            graph[u, vex] -= flow_path
            
            if(vex, u) not in graph:
                graph[vex, u] = flow_path
            else:
                graph[vex, u] += flow_path
            vex = parent[vex]
    total_edges = 0
    for edge in edge_lengths:
        total_edges = total_edges + edges
    mean_length = total_edges/total_augmenting_Paths if total_augmenting_Paths > 0 else 0
    return  max_flow_SAP, total_augmenting_Paths, mean_length

"""
    Method name: get_sink
    A helper method to determine the final sink node. The sink node is the end node of the longest acyclic path from the source. 
    Parameters: distance(dictionary)
    Returns: sink - a (u,v) tuple. 
"""
def get_sink(distance, parent):
    max_distance = max(distance.values())

    sink = [node for node, distance in distance.items() if distance == max_distance][0]
    
    ## the sink selected as the node with the longest acylic path from the source. 
    return sink, max_distance

"""
    Method name: breadth_first_search
    A method that does Breadth First Search with specified Source node.
    An helper method for determining the sink (node at the end of longest acyclic path from source)
    It calculates the distance and parent for each node, and return this result

    Parameters: List of Vertices, Edges(set), Capacity (Dictionary (u,v):capacity), adjacency list, source, and sink set to 0 initially during method call
    Returns: distance for each node from the source and their parents. 
"""
def breadth_first_search(V, E, capacities, adjlist, source):
    
    visited = set() 
    queue = deque([source])

    distance = {node: float('inf') for node in V}
    parent = {node: None for node in V}

    # Initialization
    visited.add(source)
    queue.append(source)
    distance[source] = 0
    parent[source] = None

    while queue:
        node = queue.popleft()
        ##print(node, end=" ")

        for neighbours in adjlist[node]:
            if neighbours not in visited:
                visited.add(neighbours)
                queue.append(neighbours)

                distance[neighbours] = distance[node] + 1
                parent[neighbours] = node
    
    return distance, parent


"""
This method is NOT PART OF THE PROJECT.
This method uses external libraries ONLY FOR VISUALIZATION OF THE GRAPH, and adds nothing else in solving the
problem stated in the project.
"""
## Delete this. This is just for testing.
def visualize_graph(V, E, capacities, source, sink):
    # Create a scatter plot for vertices
    for vertex in V:
        plt.scatter(*vertex, color='blue', marker='o')
        #plt.text(vertex[0], vertex[1], f'{vertex}', fontsize=8, ha='right')

    # Plot directed edges and their capacities
    for edge in E:
        plt.arrow(*edge[0], edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], head_width=0.015, head_length=0.015, fc='red', ec='red')
        #plt.text((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2, f'{capacities[edge]}', fontsize=8, color='green', ha='right')

    # Highlight source and sink nodes
    plt.scatter(*source, color='yellow', marker='s', label='Source')
    plt.scatter(*sink, color='purple', marker='s', label='Sink')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Randomly Generated Source-Sink Graph')
    plt.legend()
    plt.show()

"""
    Method name: GenerateSinkSourceGraph
    The main method that generates a graph, randomly assigning a source. 
    The vertices is a list of nodes (x, y) (where x and y are coordinates)
    Edges is stored in a set (as (u,v))
    Capacities is a dicitionary that stores {Edge: Capacity value}
    adjacency_list is also a dictionary storing {Vertex: [list of adj vertices]}.
    This method follows the psuedo code given in the project description, and also make sure to hold anti-parallel edge property.
"""

def GenerateSinkSourceGraph(n, r, upperCap):
    Vertices = []  # List of nodes
    E = set()  # Set of edges, will be added as tuple (x,y)
    capacities = {}  # Dictionary to store edge capacities
    adjacency_list = {}

    for i in range(n):
        #node = (random.uniform(0, 1), random.uniform(0, 1))
        node = (round(random.uniform(0, 1),4), round(random.uniform(0, 1),4))
        Vertices.append(node)
    for v in Vertices:
        adjacency_list[v] = []

    for u in Vertices:
        for v in Vertices:
            if u != v and (u[0] - v[0])**2 + (u[1] - v[1])**2 <= r**2:
                rand = random.uniform(0, 1)
                if rand < 0.5:
                    edge1 = (u, v)
                    edge2 = (v, u)
                    # doubling checking for parallel edge
                    if (edge1 not in E) and (edge2 not in E):
                        E.add(edge1)
                        if u in adjacency_list:
                            adjacency_list[u].append(v)
                        else:
                            adjacency_list[u] = [v]
                        #capacities[edge] = round(random.uniform(1,upperCap),4)
                else:
                    edge1 = (v, u)
                    edge2 = (u, v)
                    if (edge1 not in E) and (edge2 not in E):
                        E.add(edge1)
                        if v in adjacency_list:
                            adjacency_list[v].append(u)
                        else:
                            adjacency_list[v] = [u]
                        #capacities[edge] = round(random.uniform(1,upperCap),4)
    
    for edge in E:
        capacities[edge]= int(round(random.uniform(1, upperCap),4))

    # Randomly select source and sink nodes
    source = random.choice(Vertices)

    return Vertices, E, capacities, source, adjacency_list

def save_graph_data(V,E,capacities,adjlist,source,sink, n, r, upperCap,max_distance,total_edges_of_Graph, graph_no):

    with open(f'sim_val_{graph_no}_meta_info.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_x_coordinate','source_y_coordinate','sink_x_coordinate','sink_y_coordinate','n','r','uppercap', 'longest_pathToSink','total_edges'])
        writer.writerow([source[0], source[1],sink[0],sink[1], n, r, upperCap, max_distance, total_edges_of_Graph])
        print("meta info saved")

    with open(f'sim_val_{graph_no}_vertices.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x-coordinate','y-coordinate'])
        for v in V:
            writer.writerow(v)
    
    with open(f'sim_val_{graph_no}_edges.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['v1_x', 'v1_y', 'v2_x', 'v2_y'])
        for edge in E:
            writer.writerow([edge[0][0], edge[0][1], edge[1][0],edge[1][1]])
    
    with open(f'sim_val_{graph_no}_capacities.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['v1_x', 'v1_y', 'v2_x', 'v2_y', 'capacity'])
        for edge, capacity in capacities.items():
            writer.writerow([edge[0][0], edge[0][1], edge[1][0],edge[1][1], capacity])
    
    with open(f'sim_val_{graph_no}_adjlist.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['vertex_x', 'vertex_y', 'pairs of vertices coordinates'])  # Header
        for vertex, neighbors in adjlist.items():
            writer.writerow([vertex[0], vertex[1], *[neighbor for n in neighbors for neighbor in n]])
    print("Info saved")

def read_data(file_path1,file_path2,file_path3, file_path4, file_path5):
    with open(file_path1, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            source = (float(row[0]),float(row[1]))
            sink = (float(row[2])), float(row[3])
            n = (float(row[4]))
            r= (float(row[5]))
            upperCap = (float(row[6]))
            maxdistance = row[7]
            if maxdistance == 'Inf':
                maxdist = 18
            else:
                try:
                    maxdist = float(maxdistance)
                except ValueError:
                    maxdist = 18
            total_edges = (float(row[8]))
            
    # reading vertices
    # print(f'Reading data for vertices from the File')
    V = []
    with open(file_path2, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            x,y = map(float,row)
            V.append((x,y))


    E = set()
    with open(file_path3, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            E.add((u,v))

    capacities = {}
    with open(file_path4, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            u = (float(row[0]), float(row[1]))
            v = (float(row[2]), float(row[3]))
            capacity = int(row[4])
            capacities[(u, v)] = capacity

    adjlist = {}
    with open(file_path5, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)
        for row in reader:
            vertex = (float(row[0]), float(row[1]))
            adjacent_vertices = [(float(row[i]), float(row[i+1])) for i in range(2, len(row), 2)]
            adjlist[vertex] = adjacent_vertices

    return source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges



"""
The code block below is for setting the values and to run the program to generate random source-sink graph


"""

def main():
    # Checking first if the graph is generated data for all simulation values are present in the home directory
    simulation_values = [[100,0.2,2],[200,0.2,2],[100,0.3,2],[200,0.3,2],[100,0.2,50],[200,0.2,50],[100,0.3,50],[200,0.3,50]]
    graph_exist = True
    for graph_no in range (1,9):
        for type in ["vertices","edges","capacities","adjlist","meta_info"]:
            file_path = f'sim_val_{graph_no}_{type}.csv'
            if os.path.exists(file_path):
                continue
                #print(f"{file_path} File checked")
            else:
                print("Some or all files missing, regenerating graph and storing the data")
                print(f"file missing : {file_path}")
                graph_exist = False
                break  
    
    # make the following line to True to generate Fresh graph data. 
    #graph_exist = False 
    if graph_exist == False:
        simulation_values = [[100,0.2,2],[200,0.2,2],[100,0.3,2],[200,0.3,2],[100,0.2,50],[200,0.2,50],[100,0.3,50],[200,0.3,50]]
        #simulation_values = [[100,0.2,50]]
        graph_no = 0
        for sim_val in simulation_values:
            graph_no+=1
            n, r, upperCap = sim_val[0],sim_val[1], sim_val[2]
            V, E, capacities, source, adjlist = GenerateSinkSourceGraph(n, r, upperCap)

            
            
            distance, parent = breadth_first_search(V,E,capacities,adjlist,source)
            sink, max_distance = get_sink(distance, parent)
            visualize_graph(V, E, capacities, source, sink)
            max_distance = distance
            total_edges_of_Graph = len(E)
            save_graph_data(V,E,capacities,adjlist,source,sink, n, r, upperCap, max_distance,total_edges_of_Graph, graph_no)
            
    # if graph_exist == True, then it will read from files currently existing in the current working directory. 
    # Kindly note that files are saved in the current workign directory. If the files are not saved in the current working directory, it can throw error
    if graph_exist==True:

        for graph_no in range(1, 9):
            meta_info_path = f'sim_val_{graph_no}_meta_info.csv'
            vertices_path = f'sim_val_{graph_no}_vertices.csv'
            edges_path = f'sim_val_{graph_no}_edges.csv'
            capacities_path = f'sim_val_{graph_no}_capacities.csv'
            adjlist_path = f'sim_val_{graph_no}_adjlist.csv'

            source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
                meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path)

            print(f'\nThe source is {source} and the sink is {sink}')

            # Run Shortest Augmenting Path (SAP) algorithm
            max_flow_SAP, TAP_SAP, ml_Sap = ford_fulkerson(capacities, source, sink)
            MPL = ml_Sap/maxdist
            print(f"The maximum possible flow for SAP (graph {graph_no}) is {max_flow_SAP}")
            print(f"total augmenting paths for SAP for graph {graph_no} is {TAP_SAP}")
            print(f"Mean Length for SAP for graph {graph_no} is {ml_Sap}")
            print(f"MPL for graph {graph_no} is {MPL}")
            print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")


            source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
                meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path)

            print(f'\nThe source is {source} and the sink is {sink}')

            
            max_flow_DFS_like, TAP_DFS_like, ml_DFS_like = ford_fulkerson_DFS_like(capacities, source, sink, adjlist)
            MPL = ml_DFS_like/maxdist
            print(f"The maximum possible flow for DFS (graph {graph_no}) is {max_flow_DFS_like}")
            print(f"total augmenting paths for DFS-like for graph {graph_no} is {TAP_DFS_like}")
            print(f"Mean Length for DFS-like graph {graph_no} is {ml_DFS_like}")
            print(f"MPL for graph {graph_no} is {MPL}")
            print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")
            

            source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
                meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path)

            
            print(f'\nThe source is {source} and the sink is {sink}')
            max_flow_maxcap, TAP_maxCap, ml_maxCap = ford_fulkerson_max_capacity(capacities, source, sink, adjlist)
            MPL = ml_maxCap/maxdist
            print(f"The maximum possible flow for maxCap (graph {graph_no}) is {max_flow_maxcap}")
            print(f"total augmenting paths for maxCap for graph {graph_no} is {TAP_maxCap}")
            print(f"Mean Length for maxCap graph {graph_no} is {ml_maxCap}")
            print(f"MPL for graph {graph_no} is {MPL}")
            print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")


            source, sink, n, r, upperCap, V, E, capacities, adjlist, maxdist, total_edges_of_Graph = read_data(
                meta_info_path, vertices_path, edges_path, capacities_path, adjlist_path)

            print(f'\nThe source is {source} and the sink is {sink}')

            # Run Shortest Augmenting Path (SAP) algorithm
            max_flow_random, TAP_random, ml_random = ford_fulkerson_random(capacities, source, sink, adjlist)
            MPL = ml_random/maxdist
            print(f"The maximum possible flow for random (graph {graph_no}) is {max_flow_random}")
            print(f"total augmenting paths for random for graph {graph_no} is {TAP_random}")
            print(f"Mean Length for random graph {graph_no} is {ml_random}")
            print(f"MPL for graph {graph_no} is {MPL}")
            print(f"Total edges of graph {graph_no} is {total_edges_of_Graph}")


            


        
        

    
  

if __name__ == "__main__":
    main()



'''
n=100,r=0.2,upperCap=2 2. 
n=200,r=0.2,upperCap=2 3. 
n=100,r=0.3,upperCap=2 4. 
n=200,r=0.3,upperCap=2 5. 
n=100,r=0.2,upperCap=50 6. 
n=200,r=0.2,upperCap=50 7. 
n=100,r=0.3,upperCap=50 8. 
n=200,r=0.3,upperCap=50
'''
