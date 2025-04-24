# 3-SAT to Knapsack Reduction

## Theoretical Foundation

### Overview
The reduction from 3-SAT to Knapsack is a polynomial-time transformation that proves Knapsack is NP-complete. This reduction is particularly interesting because it shows how a logical problem (3-SAT) can be transformed into an optimization problem (Knapsack).

### Proof of Correctness

1. **Polynomial Time**:
   - Creating items: O(n) where n is number of variables
   - Computing target values: O(n)
   - Total transformation time: O(n)

2. **Equivalence**:
   - If 3-SAT is satisfiable ⟺ Knapsack has solution with value V
   - Solution mapping preserves satisfiability

### Key Properties of the Reduction

1. **Binary Choice Property**:
   ```
   For each variable xᵢ:
   - Must choose exactly one of (xᵢ, ¬xᵢ)
   - Ensured by weight/value being 2ⁱ
   ```

2. **Clause Satisfaction**:
   ```
   Each clause (lᵢ ∨ lⱼ ∨ lₖ) is satisfied if:
   - At least one literal is true
   - Corresponding item is selected in knapsack
   ```

## Example Reduction

### Starting 3-SAT Formula:
```
φ = (x₁ ∨ x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₃)
```

### Step-by-Step Transformation:

1. **Create Items**:
   ```
   For x₁: (2¹, 2¹) and (2¹, 2¹)
   For x₂: (2², 2²) and (2², 2²)
   For x₃: (2³, 2³) and (2³, 2³)
   ```

2. **Set Targets**:
   ```
   Target Weight = 2¹ + 2² + 2³ = 14
   Target Value = 14
   ```

3. **Solution Mapping**:
   ```
   If Knapsack selects: x₁, x₂, ¬x₃
   Then 3-SAT assignment: x₁=true, x₂=true, x₃=false
   ```

## Worked Example

### Input:
```python
clauses = [
    [("x1", True), ("x2", True), ("x3", False)],   # x₁ ∨ x₂ ∨ ¬x₃
    [("x1", False), ("x2", True), ("x3", True)]    # ¬x₁ ∨ x₂ ∨ x₃
]
```

### Generated Knapsack Instance:
```
Items:
1. x₁:  weight=2, value=2
2. ¬x₁: weight=2, value=2
3. x₂:  weight=4, value=4
4. ¬x₂: weight=4, value=4
5. x₃:  weight=8, value=8
6. ¬x₃: weight=8, value=8

Target Weight = 14
Target Value = 14
```

### Solution Process:

1. **Knapsack Solution**:
   ```
   Selected items:
   - x₁ (2,2)
   - x₂ (4,4)
   - ¬x₃ (8,8)
   Total: weight=14, value=14
   ```

2. **3-SAT Assignment**:
   ```
   x₁ = true
   x₂ = true
   x₃ = false
   ```

3. **Verification**:
   ```
   Clause 1: (true ∨ true ∨ true) = true
   Clause 2: (false ∨ true ∨ false) = true
   Therefore, formula is satisfied
   ```

## Why This Reduction Works

1. **Uniqueness Property**:
   - Powers of 2 ensure unique combinations
   - Can't mix-and-match to get same sum

2. **Consistency**:
   - Can't pick both xᵢ and ¬xᵢ (would exceed target)
   - Must pick one (to reach target)

3. **Satisfiability Preservation**:
   ```
   3-SAT satisfiable
   ⟺ Valid variable assignment exists
   ⟺ One literal per variable can be chosen
   ⟺ Knapsack solution with target value exists
   ```

## Implications

1. **Complexity Theory**:
   - Proves Knapsack is NP-hard
   - Since Knapsack ∈ NP, proves it's NP-complete

2. **Practical Impact**:
   - Can convert 3-SAT solvers to Knapsack solvers
   - Shows why Knapsack is hard to solve

3. **Optimization vs Decision**:
   - Shows relationship between:
     - Decision problems (3-SAT)
     - Optimization problems (Knapsack)

# Knapsack Problem

## Problem Variants

### 1. 0/1 Knapsack
- Each item can be taken or left
- Items cannot be divided
- NP-hard optimization problem
- Classic dynamic programming example

### 2. Fractional Knapsack
- Items can be divided
- Solvable in polynomial time
- Greedy algorithm optimal
- Also called continuous knapsack

### 3. Bounded Knapsack
- Limited number of each item
- Generalizes 0/1 knapsack
- Similar dynamic programming solution
- Applications in resource allocation

## Algorithms

### 1. Dynamic Programming Solution (0/1 Knapsack)
```python
def knapsack_01(values, weights, capacity):
    """
    Solve 0/1 knapsack using dynamic programming
    Time: O(nW), Space: O(nW)
    where n is number of items, W is capacity
    """
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w],
                              dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]

def reconstruct_solution(dp, weights, capacity):
    """
    Reconstruct items chosen in optimal solution
    """
    n = len(weights)
    w = capacity
    items = []

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1)
            w -= weights[i-1]

    return items[::-1]
```

### 2. Memory-Efficient Version
```python
def knapsack_01_efficient(values, weights, capacity):
    """
    Space-efficient version using only O(W) space
    """
    n = len(values)
    dp = [0] * (capacity + 1)

    for i in range(n):
        for w in range(capacity, weights[i]-1, -1):
            dp[w] = max(dp[w],
                       dp[w-weights[i]] + values[i])

    return dp[capacity]
```

### 3. Branch and Bound Solution
```python
def knapsack_branch_bound(values, weights, capacity):
    """
    Branch and bound solution for 0/1 knapsack
    Uses upper bound from fractional knapsack
    """
    n = len(values)
    best_value = 0
    best_solution = []

    def bound(k, value, weight, taken):
        """Compute upper bound for remaining capacity"""
        if weight >= capacity:
            return 0

        bound_value = value
        j = k
        totweight = weight

        while j < n and totweight + weights[j] <= capacity:
            bound_value += values[j]
            totweight += weights[j]
            j += 1

        if j < n:
            bound_value += (capacity - totweight) * \
                          (values[j] / weights[j])

        return bound_value

    def branch(k, value, weight, taken):
        nonlocal best_value, best_solution

        if weight <= capacity and value > best_value:
            best_value = value
            best_solution = taken[:]

        if k >= n:
            return

        if bound(k, value, weight, taken) <= best_value:
            return

        # Include item k
        taken.append(k)
        branch(k + 1,
               value + values[k],
               weight + weights[k],
               taken)
        taken.pop()

        # Exclude item k
        branch(k + 1, value, weight, taken)

    branch(0, 0, 0, [])
    return best_value, best_solution
```

### 4. Approximation Scheme
```python
def fptas_knapsack(values, weights, capacity, epsilon):
    """
    Fully Polynomial Time Approximation Scheme
    Guarantees (1-ε) optimal solution
    Running time: O(n²/ε)
    """
    n = len(values)
    P = max(values)  # Maximum value
    K = (epsilon * P) / n  # Scaling factor

    # Scale values
    scaled_values = [int(v/K) for v in values]

    # Solve with scaled values
    dp = knapsack_01(scaled_values, weights, capacity)

    # Return approximate solution
    return dp * K
```

## Applications

### 1. Resource Allocation
- Budget allocation
- Project selection
- Investment portfolio

### 2. Logistics
- Cargo loading
- Vehicle packing
- Warehouse storage

### 3. Computing
- Memory management
- Task scheduling
- Cloud resource allocation

## Implementation Considerations

### 1. Data Structures
```python
class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight  # For fractional

class Knapsack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.total_value = 0
        self.total_weight = 0
```

### 2. Optimization Techniques
- Sort by value/weight ratio
- Use binary representation
- Implement early stopping

### 3. Memory Management
- Rolling arrays
- Bit manipulation
- Sparse table techniques

## Special Cases

### 1. Equal Weights
- Reduces to selection problem
- Can be solved in O(n)
- Sort by value and take best fitting

### 2. Equal Values
- Minimize weight for given value
- Can be solved greedily
- Important in bin packing

### 3. Small Numbers
- Pseudo-polynomial time
- Exact solutions practical
- Use standard dynamic programming

## Research Directions

### 1. Online Algorithms
- Streaming data
- Real-time decisions
- Limited information

### 2. Parallel Approaches
- GPU acceleration
- Distributed algorithms
- Multi-core optimization

### 3. Machine Learning
- Learning good heuristics
- Neural network solutions
- Hybrid approaches

## Best Practices

### 1. Problem Analysis
- Identify constraints
- Consider special cases
- Choose appropriate variant

### 2. Algorithm Selection
- Consider input size
- Memory constraints
- Accuracy requirements

### 3. Testing
- Generate test cases
- Verify optimality
- Measure performance

## Future Trends

### 1. Quantum Computing
- Quantum algorithms
- Superposition-based solutions
- Hybrid classical-quantum

### 2. Bio-inspired Approaches
- Genetic algorithms
- Ant colony optimization
- Particle swarm optimization

### 3. Applications
- Smart logistics
- Resource optimization
- Financial planning