# Hamiltonian Cycle Problem

## Definition
A Hamiltonian cycle in a graph is a cycle that visits each vertex exactly once and returns to the starting vertex. The Hamiltonian cycle problem asks whether such a cycle exists in a given graph.

## NP-Completeness
The Hamiltonian cycle problem is NP-complete, proven through reduction from the 3-SAT problem.

### Reduction from 3-SAT
- For each variable x, create two vertices (x and ¬x)
- For each clause, create a gadget ensuring satisfaction
- The graph has a Hamiltonian cycle iff the formula is satisfiable

## Variants

### 1. Hamiltonian Path
- Find path visiting each vertex exactly once
- No need to return to start
- Also NP-complete
- Often easier in practice

### 2. Directed Hamiltonian Cycle
- Graph edges have direction
- Still NP-complete
- Generally harder than undirected version
- Important in DNA sequencing

### 3. Geometric Hamiltonian Cycle
- Vertices are points in plane
- Edges determined by distance
- Special cases may be easier
- Applications in robotics

## Algorithms

### 1. Backtracking Solution
```python
def hamiltonian_cycle(graph):
    """
    Find Hamiltonian cycle using backtracking
    graph: adjacency list representation
    """
    n = len(graph)
    path = [0]  # Start with vertex 0
    visited = {0}

    def backtrack():
        if len(path) == n and path[0] in graph[path[-1]]:
            return True

        for next_vertex in graph[path[-1]]:
            if next_vertex not in visited:
                visited.add(next_vertex)
                path.append(next_vertex)

                if backtrack():
                    return True

                visited.remove(next_vertex)
                path.pop()

        return False

    if backtrack():
        path.append(path[0])  # Complete the cycle
        return path
    return None
```

### 2. Dynamic Programming Approach
```python
def held_karp(graph):
    """
    Held-Karp algorithm for Hamiltonian cycle
    O(n²2ⁿ) time, O(n2ⁿ) space
    """
    n = len(graph)
    # dp[S][v] = shortest path visiting all vertices in S
    # ending at vertex v
    dp = {}

    def solve(S, v):
        if (S, v) in dp:
            return dp[(S, v)]

        if len(S) == 1:
            return 0 if v == 0 else float('inf')

        min_cost = float('inf')
        S_without_v = S - {v}

        for u in S_without_v:
            if graph[u][v]:  # If edge exists
                cost = solve(S_without_v, u) + 1
                min_cost = min(min_cost, cost)

        dp[(S, v)] = min_cost
        return min_cost

    # Try to find cycle starting/ending at vertex 0
    all_vertices = frozenset(range(n))
    if solve(all_vertices, 0) < float('inf'):
        return reconstruct_path(dp, graph)
    return None
```

### 3. Approximation Algorithm
```python
def christofides_approx(graph):
    """
    Christofides algorithm for metric graphs
    1.5-approximation for metric TSP
    """
    # 1. Find minimum spanning tree
    mst = minimum_spanning_tree(graph)

    # 2. Find odd-degree vertices
    odd_vertices = get_odd_degree_vertices(mst)

    # 3. Find minimum perfect matching
    matching = minimum_perfect_matching(odd_vertices)

    # 4. Combine MST and matching
    multigraph = combine_graphs(mst, matching)

    # 5. Find Eulerian circuit
    circuit = find_eulerian_circuit(multigraph)

    # 6. Convert to Hamiltonian cycle (shortcut)
    return shortcut_to_hamiltonian(circuit)
```

## Special Cases

### 1. Bipartite Graphs
- No odd-length cycles possible
- Hamiltonian cycle exists only if parts equal
- Can be checked efficiently

### 2. Planar Graphs
- Still NP-complete
- Some efficient special cases
- Important in geographic applications

### 3. Dense Graphs
- Dirac's theorem: Degree ≥ n/2 guarantees cycle
- Ore's theorem: Sum of non-adjacent degrees ≥ n
- Can be checked in polynomial time

## Applications

### 1. Transportation
- Vehicle routing
- Delivery scheduling
- Tour planning

### 2. Genetics
- DNA fragment assembly
- Genome sequencing
- Protein folding

### 3. Circuit Design
- PCB drilling
- VLSI design
- Wire routing

## Implementation Considerations

### 1. Graph Representation
```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def degree(self, v):
        return len(self.adj[v])

    def has_edge(self, u, v):
        return v in self.adj[u]
```

### 2. Optimization Techniques
- Vertex ordering heuristics
- Early pruning
- Path compression

### 3. Preprocessing
- Degree checking
- Connectivity testing
- Simple case detection

## Research Directions

### 1. Parameterized Algorithms
- Fixed-parameter tractability
- Tree-width parameters
- Structural parameters

### 2. Approximation Schemes
- Better approximation ratios
- Special graph classes
- Practical heuristics

### 3. Quantum Approaches
- Quantum annealing
- Adiabatic optimization
- Hybrid algorithms

## Best Practices

### 1. Algorithm Selection
- Consider graph properties
- Problem size analysis
- Resource constraints

### 2. Implementation
- Efficient data structures
- Memory management
- Error handling

### 3. Testing
- Generate test cases
- Verify solutions
- Performance profiling

## Future Trends

### 1. Parallel Computing
- Distributed algorithms
- GPU acceleration
- Cloud computing

### 2. Machine Learning
- Learning heuristics
- Neural networks
- Hybrid approaches

### 3. Applications
- Autonomous vehicles
- Network design
- Quantum computing
