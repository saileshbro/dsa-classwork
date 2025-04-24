"""
Hamiltonian Cycle Problem Implementation
This module implements algorithms for finding Hamiltonian cycles in graphs with text-based visualization.
"""

from typing import Dict, List, Set
import time

def print_graph(graph: Dict[int, List[int]]) -> None:
    """Print a text representation of the graph."""
    print("\nGraph Structure:")
    print("---------------")
    for vertex in sorted(graph.keys()):
        neighbors = sorted(graph[vertex])
        print(f"Node {vertex} → {' '.join(map(str, neighbors))}")
    print("---------------")

def visualize_path(graph: Dict[int, List[int]], path: List[int], visited: Set[int]) -> None:
    """
    Print a text visualization of the current path and remaining vertices.

    Args:
        graph: The graph as an adjacency list
        path: Current path being explored
        visited: Set of visited vertices
    """
    print("\nCurrent State:")
    print("-------------")
    # Show current path
    if path:
        path_str = " → ".join(map(str, path))
        print(f"Path: {path_str}")
    else:
        print("Path: (empty)")

    # Show available vertices
    remaining = sorted(set(graph.keys()) - visited)
    if remaining:
        print(f"Remaining vertices: {' '.join(map(str, remaining))}")
    else:
        print("Remaining vertices: (none)")

    # Show if current path can be closed
    if len(path) > 1 and path[0] in graph[path[-1]]:
        print("Can close cycle: Yes")
    else:
        print("Can close cycle: No")
    print("-------------")

def find_hamiltonian_cycles(graph: Dict[int, List[int]], verbose: bool = True) -> List[List[int]]:
    """
    Find all Hamiltonian cycles in the graph with step-by-step visualization.

    Args:
        graph: The graph represented as an adjacency list
        verbose: Whether to print detailed steps

    Returns:
        List of all Hamiltonian cycles found
    """
    cycles = []
    n = len(graph)

    def backtrack(path: List[int], visited: Set[int]) -> None:
        if verbose:
            print(f"\nExploring from vertex {path[-1]}")
            visualize_path(graph, path, visited)
            time.sleep(0.5)  # Pause to make the steps visible

        if len(path) == n:
            if path[0] in graph[path[-1]]:  # Can form a cycle
                cycle = path + [path[0]]
                cycles.append(cycle)
                if verbose:
                    print("\nFound Hamiltonian Cycle!")
                    print(f"Cycle: {' → '.join(map(str, cycle))}")
            return

        current = path[-1]
        for next_vertex in graph[current]:
            if next_vertex not in visited:
                visited.add(next_vertex)
                path.append(next_vertex)
                backtrack(path, visited)
                path.pop()
                visited.remove(next_vertex)

    # Start from each vertex
    for start in graph:
        if verbose:
            print(f"\nStarting from vertex {start}")
        backtrack([start], {start})

    return cycles

def print_solution_summary(cycles: List[List[int]]) -> None:
    """Print a summary of all found Hamiltonian cycles."""
    print("\n=== Solution Summary ===")
    if not cycles:
        print("No Hamiltonian cycles found!")
        return

    print(f"Found {len(cycles)} Hamiltonian cycle(s):")
    for i, cycle in enumerate(cycles, 1):
        print(f"\nCycle {i}:")
        print("Path: " + " → ".join(map(str, cycle)))
        print(f"Length: {len(cycle)-1} edges")  # -1 because last vertex is same as first

if __name__ == "__main__":
    # Example graph (same as before)
    graph = {
        1: [2, 3, 4],
        2: [1, 3, 4],
        3: [1, 2, 4, 5],
        4: [1, 2, 3, 5],
        5: [3, 4]
    }

    print("Finding Hamiltonian Cycles")
    print("=========================")
    print_graph(graph)

    print("\nSearching for cycles...")
    cycles = find_hamiltonian_cycles(graph)

    print_solution_summary(cycles)
