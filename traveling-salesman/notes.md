# Traveling Salesman Problem (TSP)

## Problem Definition

The Traveling Salesman Problem (TSP) is a fundamental combinatorial optimization problem that asks:
"Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?"

### Mathematical Formulation

Given:
- A complete weighted graph G = (V, E)
- Set of vertices V representing cities
- Set of edges E with weights w(e) representing distances
- |V| = n cities

Find:
- A Hamiltonian cycle (tour) of minimum total weight

### Properties
- NP-hard optimization problem
- NP-complete decision version
- Symmetric vs. Asymmetric variants
- Triangle inequality variant

## Solution Approaches

### 1. Exact Algorithms

#### A. Dynamic Programming (Held-Karp)
```python
# Complexity: O(n²2ⁿ)
# Space: O(n2ⁿ)
def held_karp(distances):
    n = len(distances)
    # dp[S][i] = min cost path visiting all vertices in S, ending at i
    dp = {}

    # Base cases
    for i in range(1, n):
        dp[frozenset([0, i]), i] = distances[0][i]

    # Iterate over subsets of increasing size
    for size in range(3, n + 1):
        for subset in itertools.combinations(range(1, n), size - 1):
            S = frozenset([0] + list(subset))
            for i in subset:
                dp[S, i] = min(
                    dp[S - {i}, j] + distances[j][i]
                    for j in S - {i}
                )

    # Return optimal tour cost
    return min(
        dp[frozenset(range(n)), i] + distances[i][0]
        for i in range(1, n)
    )
```

#### B. Branch and Bound
- Uses bounds to prune search space
- Priority queue for best-first search
- Lower bounds from MST or 1-tree

### 2. Approximation Algorithms

#### A. Christofides Algorithm (1.5-approximation)
```
For metric TSP:
1. Find MST T
2. Find minimum perfect matching M on odd-degree vertices
3. Combine T and M to get Eulerian graph
4. Shortcut Euler tour to Hamilton cycle
```

#### B. 2-approximation (for metric TSP)
```
1. Find MST
2. Double all edges
3. Find Euler tour
4. Take shortcuts
```

### 3. Heuristic Approaches

#### A. Nearest Neighbor
```python
def nearest_neighbor(distances):
    n = len(distances)
    unvisited = set(range(1, n))
    tour = [0]

    while unvisited:
        last = tour[-1]
        next_city = min(unvisited,
                       key=lambda x: distances[last][x])
        tour.append(next_city)
        unvisited.remove(next_city)

    tour.append(0)  # Return to start
    return tour
```

#### B. 2-opt Local Search
```python
def two_opt(tour, distances):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                if two_opt_improvement(tour, i, j, distances):
                    two_opt_swap(tour, i, j)
                    improved = True
    return tour
```

#### C. Simulated Annealing
```python
def simulated_annealing(distances):
    temp = INITIAL_TEMP
    tour = random_tour()
    best = tour

    while temp > FINAL_TEMP:
        neighbor = get_neighbor(tour)
        delta = tour_length(neighbor) - tour_length(tour)

        if delta < 0 or random.random() < exp(-delta/temp):
            tour = neighbor
            if tour_length(tour) < tour_length(best):
                best = tour

        temp *= COOLING_RATE
    return best
```

### 4. Meta-heuristic Approaches

#### A. Ant Colony Optimization
- Uses pheromone trails
- Combines exploration and exploitation
- Good for dynamic TSP variants

#### B. Genetic Algorithms
- Population of tours
- Crossover and mutation operators
- Selection based on fitness

## Implementation Details

### 1. Graph Representation
```python
# Adjacency Matrix
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Edge List
edges = [
    (city1, city2, distance),
    ...
]
```

### 2. Tour Representation
```python
# Sequence of cities
tour = [0, 2, 1, 3, 0]

# Permutation
perm = [2, 1, 3]  # implicit start/end
```

### 3. Data Structures
- Priority Queue for Branch & Bound
- Disjoint Set for MST
- Hash Table for Dynamic Programming

## Optimizations

### 1. Distance Calculations
- Precompute distances
- Use triangle inequality
- Exploit symmetry

### 2. Search Space Reduction
- Eliminate symmetric tours
- Use lower bounds
- Prune infeasible paths

### 3. Memory Management
- Bit representations
- Sparse storage
- State compression

## Practical Considerations

### 1. Problem Variants
- Metric TSP
- Euclidean TSP
- Asymmetric TSP
- Multiple TSP

### 2. Real-world Applications
- Logistics and routing
- Circuit board drilling
- DNA sequencing
- Vehicle scheduling

### 3. Instance Types
- Random uniform
- Clustered
- Real-world road networks
- Grid-based

## Performance Analysis

### 1. Time Complexity
```
Exact Algorithms:
- Brute Force: O(n!)
- Held-Karp: O(n²2ⁿ)
- Branch & Bound: O(n!) worst case

Approximation:
- Christofides: O(n³)
- 2-approximation: O(n²)

Heuristics:
- Nearest Neighbor: O(n²)
- 2-opt: O(n²) per iteration
- Simulated Annealing: O(iterations × n)
```

### 2. Solution Quality
```
Exact:
- Optimal solution
- Limited to ~20-30 cities

Approximation:
- Christofides: ≤ 1.5 × optimal
- MST-based: ≤ 2 × optimal

Heuristics:
- Nearest Neighbor: unbounded
- Local Search: local optimal
- Meta-heuristics: good empirical results
```

### 3. Trade-offs
- Solution quality vs. runtime
- Memory usage vs. speed
- Simplicity vs. performance

## Advanced Topics

### 1. Parallel Implementations
- Parallel local search
- Distributed meta-heuristics
- GPU acceleration

### 2. Learning Approaches
- Neural networks for TSP
- Reinforcement learning
- Learning heuristics

### 3. Online/Dynamic TSP
- Reoptimization strategies
- Rolling horizon approach
- Real-time updates

## Resources

### 1. Benchmark Instances
- TSPLIB
- National TSP
- Random Euclidean
- Clustered instances

### 2. Software Tools
- Concorde TSP Solver
- LKH heuristic
- Google OR-Tools

### 3. Further Reading
- "The Traveling Salesman Problem" by Applegate et al.
- "In Pursuit of the Traveling Salesman" by Cook
- Research papers and surveys