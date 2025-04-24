# Deterministic vs Non-deterministic Algorithms

## Deterministic Algorithms

A deterministic algorithm is one that, given a particular input, will always:
- Produce the same output
- Follow the same sequence of steps
- Have predictable resource usage (time and space)

### Characteristics:
1. **Predictability**: Same input → Same output every time
2. **Fixed Path**: Follows the same execution path for given input
3. **Reproducibility**: Results can be reproduced exactly
4. **Resource Usage**: Consistent time and space complexity

### Examples:
- Binary Search
- Quicksort (with fixed pivot selection)
- Dijkstra's Shortest Path
- Insertion Sort

## Non-deterministic Algorithms

A non-deterministic algorithm may produce different outputs or take different paths for the same input across different executions.

### Characteristics:
1. **Randomization**: Often incorporates random elements
2. **Variable Paths**: May follow different execution paths for same input
3. **Multiple Solutions**: Can produce different valid solutions
4. **Unpredictable Resource Usage**: Time and space requirements may vary

### Examples:
1. Las Vegas Algorithms
   - Always produce correct result but running time varies
   - Example: Randomized Quicksort

2. Monte Carlo Algorithms
   - May produce different results
   - Running time is fixed
   - Example: Miller-Rabin primality test

### Advantages of Non-deterministic Algorithms:
1. Often simpler than deterministic counterparts
2. Can solve problems more efficiently in practice
3. Useful for breaking worst-case scenarios
4. Better average-case performance in many situations

### Use Cases:
- **Deterministic**: Critical systems, financial calculations, sorting
- **Non-deterministic**: Pattern matching, optimization problems, cryptography

## Comparison

| Aspect | Deterministic | Non-deterministic |
|--------|--------------|-------------------|
| Output | Same every time | May vary |
| Execution Path | Fixed | Variable |
| Time Complexity | Fixed | May vary |
| Debugging | Easier | More challenging |
| Resource Usage | Predictable | Variable |

## Real-world Applications

### Deterministic Algorithms:
- Database queries
- Compiler operations
- Mathematical calculations
- Banking transactions

### Non-deterministic Algorithms:
- Genetic algorithms
- Simulated annealing
- Machine learning
- Password hashing
- Network routing

## When to Choose Which?

### Choose Deterministic When:
- Exact reproducibility is required
- Debugging needs to be straightforward
- System behavior must be predictable
- Resource constraints are strict

### Choose Non-deterministic When:
- Approximate solutions are acceptable
- Performance optimization is priority
- Problem space is too large for deterministic approach
- Breaking worst-case scenarios is important

# Practical Examples from Implementations

## N-Queens Problem
The N-Queens problem demonstrates both deterministic and non-deterministic approaches:

### Deterministic Approach (Backtracking)
- Uses systematic exploration of solution space
- Always follows the same path for same input
- Guaranteed to find solution if it exists
- Can be slower for larger board sizes

### Non-deterministic Approaches

#### Las Vegas Algorithm
- Randomly places queens and verifies solution
- Different execution paths each time
- Always produces correct solution when it finds one
- Runtime varies between executions
- Often faster than backtracking for larger boards

#### Min-Conflicts Algorithm
- Uses heuristic approach with random initial state
- Iteratively minimizes conflicts
- May produce different solutions
- Very efficient for large board sizes
- Might need restarts for optimal solutions

## Other Implemented Examples

### Knapsack Problem
- Demonstrates deterministic dynamic programming approach
- Guarantees optimal solution
- Has predictable resource usage based on input size

### Traveling Salesman Problem
- NP-hard optimization problem
- Can be solved using both approaches:
  - Deterministic: Branch and bound, dynamic programming
  - Non-deterministic: Genetic algorithms, simulated annealing

### Process Scheduling
- Shows practical application in operating systems
- Can use both deterministic (FCFS, SJF) and non-deterministic (Random) approaches
- Trade-off between predictability and performance