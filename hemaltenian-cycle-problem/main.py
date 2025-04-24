"""
hemaltenian-cycle-problem
This module implements algorithms for finding Hamiltonian cycles in graphs and visualizes them.
"""

import networkx as nx
import matplotlib.pyplot as plt

def create_and_visualize_graph(graph, cycle=None):
    # Create a new graph
    G = nx.Graph(graph)
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Create a circular layout for better visualization
    pos = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    
    # If a cycle is provided, highlight it
    if cycle:
        path_edges = list(zip(cycle[:-1], cycle[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=2)

    plt.title("Graph Visualization" if not cycle else f"Hamiltonian Cycle: {' -> '.join(map(str, cycle))}")
    plt.axis('off')
    plt.savefig(f"graph{'_cycle' if cycle else ''}.png")
    plt.close()

def find_all_hamiltonian_cycles(graph):
    n = len(graph)
    paths = []
    
    def is_valid_next(v, pos, path):
        if v not in graph[path[pos-1]]:
            return False
        if v in path:
            return False
        return True
    
    def hamiltonian_util(path, pos):
        if pos == n:
            if path[0] in graph[path[-1]]:
                complete_path = path + [path[0]]
                paths.append(complete_path[:])
            return

        for v in range(1, n+1):
            if is_valid_next(v, pos, path):
                path[pos] = v
                hamiltonian_util(path, pos + 1)
                path[pos] = -1

    path = [-1] * n
    path[0] = 1
    hamiltonian_util(path, 1)
    
    return paths

# Define the graph
graph = {
    1: [2, 3, 4],
    2: [1, 3, 4],
    3: [1, 2, 4, 5],
    4: [1, 2, 3, 5],
    5: [3, 4]
}

# Create and save the basic graph visualization
create_and_visualize_graph(graph)

# Find all Hamiltonian cycles
all_cycles = find_all_hamiltonian_cycles(graph)
print(f"Found {len(all_cycles)} Hamiltonian cycles:")

# Visualize each cycle
for i, cycle in enumerate(all_cycles, 1):
    print(f"Cycle {i}: {' -> '.join(map(str, cycle))}")
    create_and_visualize_graph(graph, cycle)
    plt.close()
