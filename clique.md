# Cliques in Graph Theory

## Definition

A clique in an undirected graph is a subset of vertices such that every two distinct vertices in the clique are adjacent (connected by an edge).

## Properties

1. **Complete Subgraph**
   - Every vertex in the clique is connected to every other vertex in the clique
   - Forms a complete subgraph of the original graph

2. **Maximal vs Maximum**
   - Maximal Clique: Cannot add any more vertices while maintaining the clique property
   - Maximum Clique: Largest possible clique in the graph

## Complexity Analysis

### Clique Decision Problem
- Input: Graph G and integer k
- Question: Does G contain a clique of size k?
- Complexity: NP-Complete

### Maximum Clique Problem
- Goal: Find the largest clique in a graph
- Complexity: NP-Hard
- One of Karp's 21 NP-complete problems

## Algorithms

### 1. Brute Force Approach
```python
def is_clique(graph, vertices):
    # Check if all vertices are connected to each other
    for i in vertices:
        for j in vertices:
            if i != j and not graph[i][j]:
                return False
    return True

def find_clique_brute_force(graph, k):
    n = len(graph)
    vertices = range(n)
    # Check all possible k-sized combinations
    for subset in combinations(vertices, k):
        if is_clique(graph, subset):
            return subset
    return None
```
Time Complexity: O(n^k)

### 2. Bron-Kerbosch Algorithm
```python
def bron_kerbosch(graph, r=None, p=None, x=None, maximal_cliques=None):
    if r is None:
        r = set()
        p = set(range(len(graph)))
        x = set()
        maximal_cliques = []

    if not p and not x:
        maximal_cliques.append(r.copy())
        return

    pivot = max((len(graph[v]) for v in p | x), default=0)

    for v in list(p):
        new_r = r | {v}
        new_p = p & set(graph[v])
        new_x = x & set(graph[v])
        bron_kerbosch(graph, new_r, new_p, new_x, maximal_cliques)
        p.remove(v)
        x.add(v)

    return maximal_cliques
```

## Applications

1. **Social Network Analysis**
   - Finding closely connected groups
   - Community detection
   - Influence analysis

2. **Biological Networks**
   - Protein interaction networks
   - Gene co-expression networks
   - Metabolic networks

3. **Data Mining**
   - Pattern recognition
   - Cluster analysis
   - Anomaly detection

## Approximation Algorithms

### 1. Greedy Approach
```python
def greedy_clique(graph):
    vertices = set(range(len(graph)))
    clique = set()

    while vertices:
        # Choose vertex with highest degree
        v = max(vertices, key=lambda x: sum(graph[x][y] for y in vertices))
        clique.add(v)
        # Keep only neighbors of v
        vertices = {u for u in vertices if graph[v][u]}

    return clique
```

### 2. Local Search
```python
def local_search_clique(graph, initial_clique):
    current = set(initial_clique)
    improved = True

    while improved:
        improved = False
        # Try adding one vertex
        for v in range(len(graph)):
            if v not in current:
                if all(graph[v][u] for u in current):
                    current.add(v)
                    improved = True
                    break

    return current
```

## Special Cases

1. **Perfect Graphs**
   - The clique number equals the chromatic number of every induced subgraph
   - Maximum clique can be found in polynomial time

2. **Planar Graphs**
   - Maximum clique size is at most 4
   - Can be found in polynomial time

3. **Bipartite Graphs**
   - Maximum clique size is 2
   - Easy to find all maximal cliques

## Real-World Constraints

1. **Large Graphs**
   - Need to use approximation algorithms
   - Consider parallel implementations
   - Focus on local search methods

2. **Dynamic Graphs**
   - Maintain clique information as graph changes
   - Use incremental algorithms
   - Consider temporal aspects

3. **Weighted Graphs**
   - Finding maximum weight cliques
   - Considering edge and vertex weights
   - Balancing multiple objectives

## Implementation Considerations

1. **Data Structures**
   - Adjacency matrix for dense graphs
   - Adjacency list for sparse graphs
   - Bit vectors for efficient set operations

2. **Optimization Techniques**
   - Vertex ordering heuristics
   - Pruning strategies
   - Early termination conditions

3. **Parallelization**
   - Divide search space
   - Parallel local search
   - Distributed algorithms

