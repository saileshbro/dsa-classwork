"""
Distance matrix for a Traveling Salesman Problem:
Cities: K (Kathmandu), B (Bhaktapur), P (Patan), N (Nagarkot)

Distance Table (in km):
+-----+-----+-----+-----+-----+
| To  | K   | B   | P   | N   |
| From|     |     |     |     |
+-----+-----+-----+-----+-----+
| K   | 0   | 250 | 40  | 390 |
| B   | 250 | 0   | 100 | 200 |
| P   | 210 | 100 | 0   | 250 |
| N   | 390 | 200 | 250 | 0   |
+-----+-----+-----+-----+-----+
"""

# Initialize the distance matrix
distance_matrix = {
    'K': {'K': 0,   'B': 250, 'P': 40,  'N': 390},
    'B': {'K': 250, 'B': 0,   'P': 100, 'N': 200},
    'P': {'K': 210, 'B': 100, 'P': 0,   'N': 250},
    'N': {'K': 390, 'B': 200, 'P': 250, 'N': 0}
}

# City names for reference
cities = ['K', 'B', 'P', 'N']
city_names = {
    'K': 'Kathmandu',
    'B': 'Bhaktapur',
    'P': 'Patan',
    'N': 'Nagarkot'
}

"""
Traveling Salesman Problem (TSP) Solver Implementation
Including exact, approximation, and heuristic approaches
"""

from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
import itertools
import random
import math
import heapq
import time
from collections import defaultdict

@dataclass
class City:
    id: int
    x: float
    y: float

class TSPSolver:
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.n = len(cities)
        self.distances = self._compute_distances()

    def _compute_distances(self) -> List[List[float]]:
        """Compute pairwise distances between cities"""
        distances = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = math.sqrt(
                    (self.cities[i].x - self.cities[j].x) ** 2 +
                    (self.cities[i].y - self.cities[j].y) ** 2
                )
                distances[i][j] = distances[j][i] = dist
        return distances

    def held_karp(self) -> Tuple[List[int], float]:
        """
        Exact solution using Held-Karp dynamic programming
        Time: O(n²2ⁿ), Space: O(n2ⁿ)
        """
        n = self.n
        # dp[(mask, pos)] = (cost, prev)
        dp = {}

        # Initialize base cases
        for i in range(1, n):
            dp[(1 | (1 << i), i)] = (self.distances[0][i], 0)

        # Iterate through all possible subsets of size [2, n-1]
        for size in range(2, n):
            for subset in itertools.combinations(range(1, n), size):
                # Add node 0 to subset
                mask = 1
                for node in subset:
                    mask |= (1 << node)

                # Try to find the shortest path ending in each node
                for last in subset:
                    # The previous mask doesn't include the last node
                    prev_mask = mask & ~(1 << last)
                    min_dist = float('inf')
                    prev_node = -1

                    # Try all possible previous nodes
                    for prev in subset:
                        if prev == last:
                            continue
                        curr_dist = dp.get((prev_mask, prev), (float('inf'), 0))[0] + self.distances[prev][last]
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            prev_node = prev

                    if prev_node != -1:
                        dp[(mask, last)] = (min_dist, prev_node)

        # Find the optimal tour back to node 0
        all_mask = (1 << n) - 1
        min_dist = float('inf')
        last_node = -1

        for last in range(1, n):
            curr_dist = dp.get((all_mask, last), (float('inf'), 0))[0] + self.distances[last][0]
            if curr_dist < min_dist:
                min_dist = curr_dist
                last_node = last

        if last_node == -1:
            return [], float('inf')

        # Reconstruct the tour
        tour = []
        curr_mask = all_mask
        curr_node = last_node

        while curr_node != -1:
            tour.append(curr_node)
            next_mask = curr_mask & ~(1 << curr_node)
            curr_node = dp.get((curr_mask, curr_node), (0, -1))[1]
            curr_mask = next_mask

        # Add start and end nodes
        tour.append(0)
        tour.reverse()

        return tour, min_dist

    def nearest_neighbor(self) -> Tuple[List[int], float]:
        """
        Nearest neighbor heuristic
        Time: O(n²)
        """
        unvisited = set(range(1, self.n))
        tour = [0]
        cost = 0.0

        while unvisited:
            last = tour[-1]
            next_city = min(unvisited,
                          key=lambda x: self.distances[last][x])
            cost += self.distances[last][next_city]
            tour.append(next_city)
            unvisited.remove(next_city)

        # Return to start
        cost += self.distances[tour[-1]][0]
        tour.append(0)
        return tour, cost

    def two_opt(self, tour: List[int]) -> Tuple[List[int], float]:
        """
        2-opt local search improvement
        Time: O(n²) per iteration
        """
        improved = True
        best_cost = self.tour_cost(tour)

        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    if self._two_opt_if_better(tour, i, j):
                        new_cost = self.tour_cost(tour)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            improved = True
                            break
                if improved:
                    break

        return tour, best_cost

    def _two_opt_if_better(self, tour: List[int], i: int, j: int) -> bool:
        """Test if 2-opt swap improves tour"""
        A, B = tour[i-1], tour[i]
        C, D = tour[j], tour[j+1]

        current = (self.distances[A][B] +
                  self.distances[C][D])
        proposed = (self.distances[A][C] +
                   self.distances[B][D])

        if proposed < current:
            # Reverse segment from i to j
            tour[i:j+1] = reversed(tour[i:j+1])
            return True
        return False

    def simulated_annealing(self,
                          initial_temp: float = 100.0,
                          final_temp: float = 0.01,
                          cooling_rate: float = 0.995,
                          iterations_per_temp: int = 100) -> Tuple[List[int], float]:
        """
        Simulated annealing metaheuristic
        """
        current_tour = list(range(self.n)) + [0]
        current_cost = self.tour_cost(current_tour)

        best_tour = current_tour[:]
        best_cost = current_cost

        temp = initial_temp

        while temp > final_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor by swapping two cities
                i, j = sorted(random.sample(range(1, self.n), 2))
                neighbor = current_tour[:]
                neighbor[i:j+1] = reversed(neighbor[i:j+1])

                # Calculate cost difference
                neighbor_cost = self.tour_cost(neighbor)
                delta = neighbor_cost - current_cost

                # Accept based on temperature
                if (delta < 0 or
                    random.random() < math.exp(-delta / temp)):
                    current_tour = neighbor
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_tour = current_tour[:]
                        best_cost = current_cost

            temp *= cooling_rate

        return best_tour, best_cost

    def christofides(self) -> Tuple[List[int], float]:
        """
        Christofides 1.5-approximation for metric TSP
        Requires NetworkX for minimum perfect matching
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX required for Christofides algorithm. "
                "Install with: pip install networkx"
            )

        # Find MST
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.distances[i][j])

        mst = nx.minimum_spanning_tree(G)

        # Find odd degree vertices
        odd_vertices = [v for v in G.nodes()
                       if len(list(mst.edges(v))) % 2 == 1]

        # Find minimum perfect matching
        H = nx.Graph()
        for i, v1 in enumerate(odd_vertices):
            for v2 in odd_vertices[i+1:]:
                H.add_edge(v1, v2, weight=self.distances[v1][v2])

        matching = nx.min_weight_matching(H)

        # Combine MST and matching
        multigraph = nx.MultiGraph()
        multigraph.add_edges_from(mst.edges())
        multigraph.add_edges_from(matching)

        # Find Eulerian circuit
        euler_circuit = list(nx.eulerian_circuit(multigraph))

        # Shortcut to get Hamilton cycle
        visited = set()
        tour = []
        for u, _ in euler_circuit:
            if u not in visited:
                tour.append(u)
                visited.add(u)
        tour.append(0)

        return tour, self.tour_cost(tour)

    def tour_cost(self, tour: List[int]) -> float:
        """Calculate total cost of tour"""
        return sum(self.distances[tour[i]][tour[i+1]]
                  for i in range(len(tour)-1))

def generate_random_instance(n: int,
                          min_coord: float = 0.0,
                          max_coord: float = 100.0) -> List[City]:
    """Generate random TSP instance"""
    return [
        City(i,
            random.uniform(min_coord, max_coord),
            random.uniform(min_coord, max_coord))
        for i in range(n)
    ]

def compare_algorithms(cities: List[City]):
    """Compare different TSP algorithms"""
    solver = TSPSolver(cities)
    results = {}

    # Small instances only for exact algorithm
    if len(cities) <= 20:
        print("\nHeld-Karp (Exact):")
        start = time.time()
        tour, cost = solver.held_karp()
        elapsed = time.time() - start
        results["Held-Karp"] = {
            "tour": tour,
            "cost": cost,
            "time": elapsed
        }
        print(f"Cost: {cost:.2f}")
        print(f"Time: {elapsed:.4f}s")

    print("\nNearest Neighbor:")
    start = time.time()
    tour, cost = solver.nearest_neighbor()
    elapsed = time.time() - start
    results["Nearest Neighbor"] = {
        "tour": tour,
        "cost": cost,
        "time": elapsed
    }
    print(f"Cost: {cost:.2f}")
    print(f"Time: {elapsed:.4f}s")

    print("\n2-opt Improvement:")
    start = time.time()
    tour, cost = solver.two_opt(tour[:])  # Improve NN tour
    elapsed = time.time() - start
    results["2-opt"] = {
        "tour": tour,
        "cost": cost,
        "time": elapsed
    }
    print(f"Cost: {cost:.2f}")
    print(f"Time: {elapsed:.4f}s")

    print("\nSimulated Annealing:")
    start = time.time()
    tour, cost = solver.simulated_annealing()
    elapsed = time.time() - start
    results["Simulated Annealing"] = {
        "tour": tour,
        "cost": cost,
        "time": elapsed
    }
    print(f"Cost: {cost:.2f}")
    print(f"Time: {elapsed:.4f}s")

    try:
        print("\nChristofides:")
        start = time.time()
        tour, cost = solver.christofides()
        elapsed = time.time() - start
        results["Christofides"] = {
            "tour": tour,
            "cost": cost,
            "time": elapsed
        }
        print(f"Cost: {cost:.2f}")
        print(f"Time: {elapsed:.4f}s")
    except ImportError as e:
        print(f"Skipping Christofides: {e}")

    return results

if __name__ == "__main__":
    # Test with different instance sizes
    for n in [10, 20, 50, 100]:
        print(f"\nTesting with {n} cities:")
        print("=" * 40)
        cities = generate_random_instance(n)
        results = compare_algorithms(cities)

        # Additional analysis for small instances
        if n <= 20 and "Held-Karp" in results:
            optimal = results["Held-Karp"]["cost"]
            print("\nApproximation Ratios:")
            for name, data in results.items():
                if name != "Held-Karp":
                    ratio = data["cost"] / optimal
                    print(f"{name}: {ratio:.3f}")