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