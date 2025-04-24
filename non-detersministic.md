# Computational Complexity Classes

## P (Polynomial Time)
- Problems that can be solved by a deterministic algorithm in polynomial time
- Solution can be found and verified in polynomial time
- Examples:
  1. Sorting algorithms (Quicksort, Mergesort) - O(n log n)
  2. Binary Search - O(log n)
  3. Matrix Multiplication - O(n³)
  4. Finding Shortest Path using Dijkstra's Algorithm - O(V² + E)

## NP (Non-deterministic Polynomial Time)
- Problems where a solution can be verified in polynomial time
- Finding a solution may take exponential time
- All P problems are in NP, but not vice versa (unless P = NP)
- Examples:
  1. Boolean Satisfiability Problem (SAT)
  2. Traveling Salesman Problem (TSP)
  3. Subset Sum Problem
  4. Graph Coloring Problem

## NP-Complete
- Hardest problems in NP
- If any NP-complete problem can be solved in polynomial time, then all NP problems can be
- A problem is NP-complete if:
  1. It is in NP
  2. Every problem in NP can be reduced to it in polynomial time
- Examples:
  1. 3-SAT (Boolean satisfiability with 3 variables per clause)
  2. Hamilton Cycle Problem
  3. Vertex Cover Problem
  4. N-Queens Problem

## NP-Hard
- Problems that are at least as hard as the hardest problems in NP
- May not be in NP themselves
- Don't necessarily have verifiable solutions in polynomial time
- Examples:
  1. Halting Problem
  2. Traveling Salesman Optimization Problem
  3. Graph Isomorphism Problem
  4. Knapsack Optimization Problem

## Relationships
```
P ⊆ NP ⊆ NP-Hard
     ∩
NP-Complete
```

## Key Differences
1. P problems: Can be both solved and verified quickly
2. NP problems: Can be verified quickly, but may not be solved quickly
3. NP-Complete problems: Hardest problems in NP
4. NP-Hard problems: At least as hard as NP-Complete problems, but may not be in NP

## Real-world Applications
- **P**: Most everyday computational tasks (sorting, searching, basic graph algorithms)
- **NP**: Resource allocation, scheduling, protein folding
- **NP-Complete**: Circuit design, network design, game theory
- **NP-Hard**: Optimization problems in logistics, artificial intelligence, machine learning

## Practical Algorithm Examples

### P Class Example: Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
Time Complexity: O(log n)

### NP Class Example: Subset Sum (Decision Version)
```python
def subset_sum_exists(numbers, target_sum):
    def backtrack(index, current_sum):
        if current_sum == target_sum:
            return True
        if index >= len(numbers) or current_sum > target_sum:
            return False

        # Include current number
        if backtrack(index + 1, current_sum + numbers[index]):
            return True

        # Exclude current number
        return backtrack(0, 0)
```
Verification: O(n), Solution Finding: O(2ⁿ)

### NP-Complete Example: Graph Coloring
```python
def is_safe(graph, colors, vertex, color):
    for adj in range(len(graph)):
        if graph[vertex][adj] and colors[adj] == color:
            return False
    return True

def graph_coloring(graph, m, vertex, colors):
    if vertex == len(graph):
        return True

    for color in range(1, m + 1):
        if is_safe(graph, colors, vertex, color):
            colors[vertex] = color
            if graph_coloring(graph, m, vertex + 1, colors):
                return True
            colors[vertex] = 0
    return False
```
Verification: O(V²), Solution Finding: O(mᵛ)

### NP-Hard Example: Traveling Salesman (Optimization Version)
```python
def tsp_optimization(graph):
    n = len(graph)
    dp = {}

    def solve(mask, pos):
        if mask == (1 << n) - 1:
            return graph[pos][0]

        key = (mask, pos)
        if key in dp:
            return dp[key]

        ans = float('inf')
        for city in range(n):
            if not (mask & (1 << city)):
                new_ans = graph[pos][city] + solve(mask | (1 << city), city)
                ans = min(ans, new_ans)

        dp[key] = ans
        return ans

    return solve(1, 0)  # Start from city 0
```
Time Complexity: O(n² × 2ⁿ)

## Las Vegas vs Monte Carlo Algorithms

### Las Vegas Algorithm Example (Quicksort)
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = random.choice(arr)  # Random pivot selection
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)
```
- Always produces correct results
- Running time is random
- Example: QuickSort with random pivot

### Monte Carlo Algorithm Example (Miller-Rabin Primality Test)
```python
def miller_rabin(n, k=5):
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False

    def check(a, d, n, s):
        x = pow(a, d, n)
        if x == 1:
            return True
        for _ in range(s - 1):
            if x == n - 1:
                return True
            x = (x * x) % n
        return x == n - 1

    s = 0
    d = n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        if not check(a, d, n, s):
            return False
    return True
```
- May produce incorrect results with small probability
- Running time is deterministic
- Example: Miller-Rabin primality testing

## Randomized Algorithms Classification

### 1. Las Vegas Algorithms (LV)
- Characteristics:
  - Always give correct results
  - Running time is random
  - May sometimes fail and report failure
- Additional Examples:
  1. Randomized Quick Select: O(n) expected time
  2. Randomized Binary Search Trees
  3. Skiplist Data Structure: O(log n) expected time

### 2. Monte Carlo Algorithms (MC)
- Characteristics:
  - Fixed running time
  - May give incorrect results with small probability
  - Never report failure
- Additional Examples:
  1. Bloom Filters: Membership testing with false positives
  2. Random Sampling
  3. Karger's Algorithm for Minimum Cut
  4. Randomized Primality Testing

### Practical Applications

#### 1. Cryptography
- Random number generation
- Key generation
- Probabilistic encryption schemes
- Digital signatures

#### 2. Distributed Systems
- Leader election algorithms
- Load balancing
- Consensus protocols
- Network protocol testing

#### 3. Machine Learning
- Random forests
- Stochastic gradient descent
- Random feature selection
- Cross-validation with random splits

#### 4. Big Data Processing
- Reservoir sampling
- Random projections
- Locality-sensitive hashing
- Approximate counting algorithms

## Trade-offs in Using Randomized Algorithms

### Advantages
1. Often simpler to implement
2. Faster average-case performance
3. Can break worst-case scenarios
4. May avoid patent issues (different approach)

### Disadvantages
1. Results may vary between runs
2. Debugging can be challenging
3. May need good random number generators
4. Not suitable for critical applications requiring deterministic results