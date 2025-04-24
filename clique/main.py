"""
Maximum Clique Problem Implementations
Including exact, approximation, and heuristic approaches
"""

from typing import List, Set, Dict
import random
import math
from collections import defaultdict
import time

class Graph:
    def __init__(self):
        self.adj_list: Dict[int, Set[int]] = defaultdict(set)
        self.num_vertices = 0

    def add_edge(self, u: int, v: int) -> None:
        self.adj_list[u].add(v)
        self.adj_list[v].add(u)
        self.num_vertices = max(self.num_vertices, u + 1, v + 1)

    def is_edge(self, u: int, v: int) -> bool:
        return v in self.adj_list[u]

def brute_force_max_clique(graph: Graph) -> Set[int]:
    """Find maximum clique using brute force approach."""
    def is_clique(vertices: Set[int]) -> bool:
        return all(graph.is_edge(u, v) for u in vertices for v in vertices if u < v)

    max_clique: Set[int] = set()
    n = graph.num_vertices

    # Try all possible vertex combinations
    for i in range(1 << n):
        current: Set[int] = {j for j in range(n) if (i & (1 << j))}
        if is_clique(current) and len(current) > len(max_clique):
            max_clique = current

    return max_clique

def bron_kerbosch(graph: Graph) -> Set[int]:
    """Find maximum clique using Bron-Kerbosch algorithm with pivoting."""
    def bk_pivot(r: Set[int], p: Set[int], x: Set[int], max_clique: List[Set[int]]) -> None:
        if not p and not x:
            if not max_clique or len(r) > len(max_clique[0]):
                max_clique[0] = r.copy()
            return

        pivot = max((len(graph.adj_list[v] & p), v) for v in p.union(x))[1]
        for v in p - graph.adj_list[pivot]:
            neighbors = graph.adj_list[v]
            bk_pivot(r | {v},
                    p & neighbors,
                    x & neighbors,
                    max_clique)
            p.remove(v)
            x.add(v)

    max_clique = [set()]
    p = set(range(graph.num_vertices))
    bk_pivot(set(), p, set(), max_clique)
    return max_clique[0]

def greedy_max_clique(graph: Graph) -> Set[int]:
    """Find a maximal clique using greedy approach."""
    # Sort vertices by degree
    vertices = sorted(range(graph.num_vertices),
                     key=lambda x: len(graph.adj_list[x]),
                     reverse=True)

    clique: Set[int] = {vertices[0]}

    for v in vertices[1:]:
        if all(graph.is_edge(v, u) for u in clique):
            clique.add(v)

    return clique

def simulated_annealing_max_clique(graph: Graph,
                                 initial_temp: float = 100.0,
                                 cooling_rate: float = 0.995,
                                 iterations_per_temp: int = 100) -> Set[int]:
    """Find maximum clique using simulated annealing."""
    def clique_size(vertices: Set[int]) -> int:
        return sum(1 for i in vertices
                  for j in vertices
                  if i < j and graph.is_edge(i, j))

    current = set(range(graph.num_vertices))
    best = current.copy()
    best_size = clique_size(best)
    temp = initial_temp

    while temp > 0.1:
        for _ in range(iterations_per_temp):
            # Generate neighbor by flipping one vertex
            neighbor = current.copy()
            if random.random() < 0.5 and neighbor:
                neighbor.remove(random.choice(list(neighbor)))
            else:
                possible_adds = set(range(graph.num_vertices)) - neighbor
                if possible_adds:
                    neighbor.add(random.choice(list(possible_adds)))

            # Calculate energy difference
            current_size = clique_size(current)
            neighbor_size = clique_size(neighbor)
            delta_e = neighbor_size - current_size

            # Accept or reject new solution
            if (delta_e > 0 or
                random.random() < math.exp(delta_e / temp)):
                current = neighbor
                if neighbor_size > best_size:
                    best = neighbor.copy()
                    best_size = neighbor_size

        temp *= cooling_rate

    return best

def generate_random_graph(n: int, p: float) -> Graph:
    """Generate random graph with n vertices and edge probability p."""
    g = Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g.add_edge(i, j)
    return g

if __name__ == "__main__":
    # Example usage and testing
    g = Graph()
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    for u, v in edges:
        g.add_edge(u, v)

    print("Brute Force:", brute_force_max_clique(g))
    print("Bron-Kerbosch:", bron_kerbosch(g))
    print("Greedy:", greedy_max_clique(g))
    print("Simulated Annealing:", simulated_annealing_max_clique(g))

    # Generate and test random graph
    random_graph = generate_random_graph(10, 0.3)
    print("\nRandom Graph Results:")
    print("Brute Force:", brute_force_max_clique(random_graph))
    print("Bron-Kerbosch:", bron_kerbosch(random_graph))
    print("Greedy:", greedy_max_clique(random_graph))
    print("Simulated Annealing:", simulated_annealing_max_clique(random_graph))
