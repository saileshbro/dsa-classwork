# Clique Problem

## Definition
A clique in an undirected graph is a subset of vertices such that every two distinct vertices in the clique are adjacent. The clique problem refers to finding cliques of various sizes in a graph.

## Problem Variants

### 1. Maximum Clique Problem
- Find the largest clique in a graph
- NP-hard optimization problem
- Decision version is NP-complete

### 2. k-Clique Problem
- Determine if graph contains clique of size k
- NP-complete for arbitrary k
- Polynomial time for fixed k

### 3. Maximal Clique Enumeration
- List all maximal cliques
- Can be exponential in size
- Important in social network analysis

## Applications

### 1. Social Network Analysis
- Community detection
- Social group identification
- Influence maximization

### 2. Bioinformatics
- Protein interaction networks
- Gene expression analysis
- Molecular structure analysis

### 3. Pattern Recognition
- Image segmentation
- Feature matching
- Pattern identification

## Algorithms

### 1. Exact Algorithms

#### Bron-Kerbosch Algorithm
```python
def bron_kerbosch(R, P, X, graph):
    """
    R: current clique being built
    P: prospective vertices
    X: excluded vertices
    """
    if not P and not X:
        # Found maximal clique
        yield R
        return

    pivot = max(P.union(X), key=lambda v: len(set(graph[v]) & P))

    for v in P.difference(graph[pivot]):
        neighbors = set(graph[v])
        yield from bron_kerbosch(
            R.union({v}),
            P.intersection(neighbors),
            X.intersection(neighbors),
            graph
        )
        P.remove(v)
        X.add(v)
```

#### Branch and Bound
```python
def branch_and_bound_clique(graph):
    """
    Find maximum clique using branch and bound
    - Uses vertex ordering by degree
    - Prunes branches that can't improve solution
    """
    n = len(graph)
    max_clique = set()

    def bound(vertices):
        return len(vertices)

    def branch(clique, candidates):
        nonlocal max_clique
        if bound(candidates) + len(clique) <= len(max_clique):
            return

        if not candidates:
            if len(clique) > len(max_clique):
                max_clique = clique.copy()
            return

        v = candidates.pop()
        # Include v
        new_candidates = {u for u in candidates
                         if graph[u][v]}
        branch(clique | {v}, new_candidates)

        # Exclude v
        branch(clique, candidates)

    initial_candidates = set(range(n))
    branch(set(), initial_candidates)
    return max_clique
```

### 2. Approximation Algorithms

#### Greedy Clique Finding
```python
def greedy_max_clique(graph):
    """
    Simple greedy approximation
    - O(V²) time complexity
    - No guarantee on approximation ratio
    """
    vertices = set(range(len(graph)))
    clique = set()

    while vertices:
        v = max(vertices,
                key=lambda x: sum(1 for u in clique
                                if graph[x][u]))
        if all(graph[v][u] for u in clique):
            clique.add(v)
        vertices.remove(v)

    return clique
```

#### Color-based Approximation
```python
def color_based_clique(graph):
    """
    Uses graph coloring for approximation
    - O(V² log V) time complexity
    - O(n/log n) approximation ratio
    """
    n = len(graph)
    colors = graph_coloring(complement_graph(graph))
    max_color_class = max(range(n),
                         key=lambda c: sum(1 for v in range(n)
                                         if colors[v] == c))
    return {v for v in range(n) if colors[v] == max_color_class}
```

## Complexity Analysis

### 1. Time Complexity
- Exact algorithms: O(3^(n/3)) (Bron-Kerbosch)
- Approximation: O(n²) to O(n³)
- Parameterized: O(2^k * n) for k-clique

### 2. Space Complexity
- Exact: O(n²) for adjacency matrix
- Approximation: O(n) to O(n²)
- Output-sensitive variations possible

## Special Cases

### 1. Perfect Graphs
- Maximum clique solvable in polynomial time
- Uses semi-definite programming
- Important in theoretical analysis

### 2. Planar Graphs
- Maximum clique size ≤ 4
- Can be found in linear time
- Used in geographic applications

### 3. Random Graphs
- Expected clique size: O(log n)
- Probabilistic analysis important
- Used in average-case analysis

## Implementation Considerations

### 1. Graph Representation
```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj_matrix = [[0] * vertices
                          for _ in range(vertices)]

    def add_edge(self, v1, v2):
        self.adj_matrix[v1][v2] = 1
        self.adj_matrix[v2][v1] = 1

    def is_clique(self, vertices):
        return all(self.adj_matrix[i][j]
                  for i, j in combinations(vertices, 2))
```

### 2. Performance Optimization
- Use bit operations for small graphs
- Implement vertex ordering heuristics
- Consider parallel implementations

### 3. Memory Management
- Use sparse representations for large graphs
- Implement iterative algorithms
- Consider external memory algorithms

## Research Directions

### 1. Approximation Algorithms
- Better approximation ratios
- New theoretical bounds
- Practical heuristics

### 2. Parallel Algorithms
- GPU implementations
- Distributed computing
- Quantum approaches

### 3. Applications
- Social network analysis
- Biological networks
- Pattern recognition

## Best Practices

### 1. Algorithm Selection
- Consider graph size and density
- Evaluate solution quality requirements
- Assess computational resources

### 2. Implementation
- Use appropriate data structures
- Implement efficient pruning
- Consider parallel processing

### 3. Testing
- Use diverse test cases
- Validate against known results
- Measure performance metrics

# Maximum Clique Problem

## Definition
A clique in an undirected graph is a subset of vertices such that every two distinct vertices in the clique are adjacent (connected by an edge). The Maximum Clique Problem (MCP) involves finding a clique of the largest possible size in a given graph.

## Complexity
- The Maximum Clique Problem is NP-hard
- The decision version ("Is there a clique of size k?") is NP-complete
- Even approximating the maximum clique size within a factor of n¹⁻ᵋ is NP-hard for any ε > 0

## Implemented Algorithms

### 1. Brute Force Algorithm
- **Approach**: Examines all possible subsets of vertices
- **Time Complexity**: O(2ⁿ * n²)
- **Guarantees**: Finds the optimal solution
- **Best Use**: Small graphs (n ≤ 20)

### 2. Bron-Kerbosch Algorithm
- **Approach**: Recursive backtracking with pivoting
- **Time Complexity**: O(3^(n/3)) worst case
- **Advantages**:
  - Much faster than brute force in practice
  - Finds all maximal cliques
  - Widely used in real applications
- **Optimizations**:
  - Pivot selection
  - Degeneracy ordering

### 3. Greedy Algorithm
- **Approach**: Builds clique by adding vertices sorted by degree
- **Time Complexity**: O(V²)
- **Properties**:
  - Fast but no approximation guarantee
  - Can be used as initial solution for other algorithms
  - May find optimal solution in special cases

### 4. Simulated Annealing
- **Approach**: Probabilistic metaheuristic
- **Features**:
  - Escapes local optima
  - Controlled by temperature parameter
  - Balance between exploration and exploitation
- **Parameters**:
  - Initial temperature
  - Cooling rate
  - Iterations per temperature

## Applications
1. Social Network Analysis
   - Finding closely connected groups
   - Community detection

2. Bioinformatics
   - Protein structure prediction
   - Gene expression analysis

3. Computer Vision
   - Feature matching
   - Image segmentation

4. Network Analysis
   - Network vulnerability assessment
   - Communication pattern detection

## Implementation Details
Our implementation in `clique_implementations.py` provides:
- Modular Graph class for graph operations
- Multiple algorithm implementations
- Experimental framework for comparison
- Random graph generator for testing

## Performance Comparison
- Brute Force: Optimal but limited to small graphs
- Bron-Kerbosch: Best for exact solutions on medium-sized graphs
- Greedy: Fastest but solution quality varies
- Simulated Annealing: Good balance of quality and speed for large graphs

## Best Practices
1. For small graphs (n ≤ 20): Use Brute Force
2. For medium graphs (20 < n ≤ 100): Use Bron-Kerbosch
3. For large graphs (n > 100): Use Simulated Annealing
4. For initial solutions: Use Greedy algorithm

## Further Research Directions
1. Parallel implementations
2. Hybrid algorithms
3. Application-specific heuristics
4. Quantum computing approaches