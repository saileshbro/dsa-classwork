# NP-Completeness and Boolean Satisfiability (SAT)

## Introduction to NP-Completeness

NP-complete problems are the hardest problems in the NP (Non-deterministic Polynomial time) class. A problem is NP-complete if:
1. It is in NP (solution can be verified in polynomial time)
2. Every other problem in NP can be reduced to it in polynomial time

The Boolean Satisfiability (SAT) problem is particularly important because it was the first problem proven to be NP-complete (Cook-Levin theorem).

## Boolean Satisfiability (SAT) Problem

### Definition
The Boolean Satisfiability Problem (SAT) asks whether there exists an assignment of boolean values (True/False) to variables that makes a given boolean formula evaluate to True.

### Why SAT is NP-Complete
1. SAT is in NP:
   - Given an assignment, we can verify if it satisfies the formula in polynomial time
   - Simply evaluate each clause with the given assignment

2. All NP problems reduce to SAT:
   - Any problem in NP can be transformed into a SAT formula
   - The transformation takes polynomial time
   - If we can solve SAT, we can solve any NP problem

### Example
Consider the formula: (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₂) ∧ (x₁ ∨ ¬x₂)

Where:
- ∨ represents OR
- ∧ represents AND
- ¬ represents NOT

Solution approach:
1. Try different assignments of True/False to x₁ and x₂
2. Check if any assignment satisfies all clauses
3. Use tree pruning to optimize search

### Implementation Details
```python
# Example clauses:
[(1, True), (2, True)]      # x1 v x2
[(1, False), (2, True)]     # ~x1 v x2
[(1, True), (2, False)]     # x1 v ~x2
```

## Implementation Example

Here's a Python implementation of a SAT solver using tree pruning:

```python
def is_satisfiable(assignment: Dict[int, bool]) -> bool:
    return all(self.evaluate_clause(clause, assignment) for clause in self.clauses)

def can_be_satisfied(assignment: Dict[int, bool]) -> bool:
    """Check if partial assignment can potentially lead to a solution"""
    for clause in self.clauses:
        clause_can_be_true = False
        for var, is_positive in clause:
            if var in assignment:
                if (is_positive and assignment[var]) or (not is_positive and not assignment[var]):
                    clause_can_be_true = True
                    break
            else:
                clause_can_be_true = True
                break
        if not clause_can_be_true:
            return False
    return True
```

Example usage with formula: (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₂) ∧ (x₁ ∨ ¬x₂) ∧ (¬x₂ ∨ x₃) ∧ (¬x₂ ∨ ¬x₃)

```python
clauses = [
    [(1, True), (2, True)],      # x1 v x2
    [(1, False), (2, True)],     # ~x1 v x2
    [(1, True), (2, False)],     # x1 v ~x2
    [(2, False), (3, True)],     # ~x2 v x3
    [(2, False), (3, False)]     # ~x2 v ~x3
]
```

The solver uses:
1. Tree pruning to eliminate branches that can't lead to solutions
2. Efficient clause evaluation for partial assignments
3. Backtracking to explore all possible assignments systematically

### Key Optimizations
- Early pruning of unsatisfiable branches
- Smart backtracking when conflicts are detected
- Efficient clause evaluation for partial assignments

### Complexity Analysis
- Worst case: O(2ⁿ) where n is number of variables
- Space complexity: O(n)
- Verification: O(m) where m is number of clauses

### Practical Applications and Implications

1. Used in:
   - Hardware verification
   - Software testing
   - AI planning systems
   - Circuit design

2. Optimization Techniques:
   - Unit propagation: Simplify formula by assigning forced values
   - Clause learning: Remember conflicts to avoid repeated work
   - Backjumping: Smart backtracking when conflicts occur

### Importance in Theory

1. Historical Significance:
   - First NP-complete problem (Cook-Levin theorem, 1971)
   - Foundation for proving other problems NP-complete

2. Reductions:
   - New problems proven NP-complete by reducing SAT to them
   - Shows universal nature of boolean logic

## 3-SAT and 3-CNF Satisfiability

### What is 3-SAT?
3-SAT (3-CNF-SAT) is a special case of the Boolean satisfiability problem where each clause in the formula contains exactly three literals. It is one of the most fundamental NP-complete problems.

### Definition
- A 3-CNF formula is a conjunction (AND) of clauses
- Each clause is a disjunction (OR) of exactly 3 literals
- Each literal is either a variable or its negation

Example:
```
(x₁ ∨ x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₄) ∧ (x₂ ∨ ¬x₃ ∨ ¬x₄)
```

### Why is 3-SAT Important?

1. **NP-Completeness**
   - 3-SAT is NP-complete
   - It's often used as a starting point for proving other problems are NP-complete
   - Simpler than general SAT but still captures the essence of NP-completeness

2. **Sweet Spot**
   - 2-SAT is solvable in polynomial time
   - 3-SAT is NP-complete
   - Shows the boundary between tractable and intractable problems

### Implementation Example

```python
def is_clause_satisfied(clause, assignment):
    """
    Check if a 3-CNF clause is satisfied by a given assignment

    Args:
        clause: tuple of (literal, is_negated) pairs
        assignment: dict mapping variables to boolean values
    """
    for var, is_negated in clause:
        value = assignment.get(var, False)
        if is_negated:
            value = not value
        if value:  # If any literal is True, clause is satisfied
            return True
    return False

def is_3sat_satisfied(formula, assignment):
    """
    Check if a 3-CNF formula is satisfied by a given assignment

    Args:
        formula: list of clauses
        assignment: dict mapping variables to boolean values
    """
    return all(is_clause_satisfied(clause, assignment) for clause in formula)

# Example usage:
formula = [
    [('x1', False), ('x2', False), ('x3', True)],   # (x₁ ∨ x₂ ∨ ¬x₃)
    [('x1', True), ('x2', False), ('x4', False)],   # (¬x₁ ∨ x₂ ∨ x₄)
    [('x2', False), ('x3', True), ('x4', True)]     # (x₂ ∨ ¬x₃ ∨ ¬x₄)
]

assignment = {'x1': True, 'x2': True, 'x3': False, 'x4': True}
is_satisfied = is_3sat_satisfied(formula, assignment)
```

### Converting General SAT to 3-SAT

We can convert any SAT formula to 3-SAT form using the following rules:

1. **For clauses with more than 3 literals:**
   ```
   (x₁ ∨ x₂ ∨ x₃ ∨ x₄) becomes
   (x₁ ∨ x₂ ∨ y) ∧ (¬y ∨ x₃ ∨ x₄)
   ```

2. **For clauses with fewer than 3 literals:**
   ```
   (x₁ ∨ x₂) becomes
   (x₁ ∨ x₂ ∨ y) ∧ (x₁ ∨ x₂ ∨ ¬y)
   ```

Here's an implementation of the conversion:

```python
def convert_to_3sat(general_clause):
    """
    Convert a general SAT clause to 3-SAT form

    Args:
        general_clause: list of literals of any length
    Returns:
        list of 3-SAT clauses
    """
    if len(general_clause) == 3:
        return [general_clause]
    elif len(general_clause) < 3:
        # Add dummy variables to make it 3-SAT
        dummy = f"y_{id(general_clause)}"
        return [
            general_clause + [(dummy, False)],
            general_clause + [(dummy, True)]
        ]
    else:
        # Break down into 3-literal clauses
        result = []
        current = general_clause[:2]
        for lit in general_clause[2:]:
            dummy = f"y_{id(lit)}"
            result.append(current + [(dummy, False)])
            current = [(dummy, True), lit]
        return result

# Example usage:
long_clause = [('x1', False), ('x2', False), ('x3', False), ('x4', False)]
three_sat_form = convert_to_3sat(long_clause)
```

### Phase Transition in 3-SAT

An interesting phenomenon in 3-SAT is the phase transition that occurs at a critical ratio of clauses to variables:

1. **Under-constrained Region (ratio < 4.27)**
   - Many satisfying assignments likely exist
   - Relatively easy to solve
   - High probability of being satisfiable

2. **Critical Region (ratio ≈ 4.27)**
   - Phase transition occurs
   - Hardest instances tend to be here
   - Equal probability of being satisfiable/unsatisfiable

3. **Over-constrained Region (ratio > 4.27)**
   - Few or no satisfying assignments
   - Can be easier to prove unsatisfiable
   - High probability of being unsatisfiable

```python
def calculate_3sat_ratio(num_clauses, num_variables):
    """
    Calculate the clause-to-variable ratio for a 3-SAT formula
    """
    return num_clauses / num_variables

def estimate_difficulty(ratio):
    """
    Estimate the difficulty of a 3-SAT instance based on its ratio
    """
    if ratio < 4.0:
        return "Likely easy (under-constrained)"
    elif 4.0 <= ratio <= 4.5:
        return "Likely hard (phase transition region)"
    else:
        return "May be easier (over-constrained)"

# Example usage:
num_clauses = 100
num_variables = 20
ratio = calculate_3sat_ratio(num_clauses, num_variables)
difficulty = estimate_difficulty(ratio)
```

### Applications of 3-SAT

1. **Circuit Design**
   - Verification of digital circuits
   - Testing for satisfiability of timing constraints

2. **AI Planning**
   - Converting planning problems to 3-SAT
   - Using SAT solvers for automated planning

3. **Cryptography**
   - Analysis of cryptographic algorithms
   - Design of secure systems

4. **Software Testing**
   - Test case generation
   - Path coverage analysis

### Common Solution Techniques

1. **DPLL Algorithm (Davis-Putnam-Logemann-Loveland)**
   ```python
   def dpll_3sat(formula, assignment=None):
       """
       Solve 3-SAT using DPLL algorithm
       """
       if assignment is None:
           assignment = {}

       # Base cases
       if not formula:
           return True, assignment
       if any(not clause for clause in formula):
           return False, None

       # Unit propagation and pure literal elimination
       # ... implementation details ...

       # Choose a variable and try both assignments
       var = choose_variable(formula)
       for value in [True, False]:
           assignment[var] = value
           new_formula = simplify_formula(formula, var, value)
           solved, result = dpll_3sat(new_formula, assignment.copy())
           if solved:
               return True, result

       return False, None
   ```

2. **Local Search Methods**
   - WalkSAT
   - Simulated Annealing
   - Genetic Algorithms

### Performance Considerations

1. **Variable Selection Heuristics**
   - VSIDS (Variable State Independent Decaying Sum)
   - DLIS (Dynamic Largest Individual Sum)
   - JW (Jeroslow-Wang)

2. **Clause Learning**
   - Conflict-Driven Clause Learning (CDCL)
   - First Unique Implication Point (1UIP)

3. **Preprocessing Techniques**
   - Variable elimination
   - Subsumption
   - Self-subsuming resolution

This deep dive into 3-SAT shows why it's a cornerstone problem in computational complexity theory and practical computer science applications.

## Reduction from 3-SAT to Knapsack

### The Reduction Process

#### Overview
The reduction from 3-SAT to Knapsack demonstrates that if we could solve Knapsack in polynomial time, we could also solve 3-SAT in polynomial time. This is a crucial part of proving that Knapsack is NP-complete.

#### Steps of Reduction

1. **Input**: A 3-SAT formula φ with n variables and m clauses
2. **Output**: A Knapsack instance that is solvable if and only if φ is satisfiable

#### The Transformation

Given a 3-SAT formula with n variables x₁, x₂, ..., xₙ and m clauses:

1. **Create Items**:
   - For each variable xᵢ, create two items representing xᵢ and ¬xᵢ
   - Each item's weight = 2ⁱ
   - Each item's value = 2ⁱ

2. **Set Target Weight**:
   - For each clause, add a number to represent satisfaction
   - Total target weight W = Σ(2ⁱ) for all variables in satisfying clauses

3. **Set Target Value**:
   - Value V must equal W for a valid solution
   - This ensures we pick exactly one of xᵢ or ¬xᵢ for each variable

Example:
```
3-SAT Formula: (x₁ ∨ x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₃)

Becomes Knapsack Instance:
Items (weight, value):
x₁:  (2¹, 2¹)    ¬x₁: (2¹, 2¹)
x₂:  (2², 2²)    ¬x₂: (2², 2²)
x₃:  (2³, 2³)    ¬x₃: (2³, 2³)

Target Weight = Target Value = 2¹ + 2² + 2³
```

### Implementation of the Reduction

Let's create a practical implementation that shows how to reduce a 3-SAT instance to a Knapsack instance:

# Boolean Satisfiability Problem (SAT)

## Overview

The Boolean Satisfiability Problem (SAT) is a fundamental problem in computer science that asks whether there exists an assignment of boolean values (True/False) to variables that makes a given boolean formula evaluate to True.

## Problem Definition

Given:
- A boolean formula F in Conjunctive Normal Form (CNF)
- A set of variables x₁, x₂, ..., xₙ

Find:
- An assignment of True/False to each variable that makes F true, or
- Determine that no such assignment exists (UNSAT)

### CNF Format
A formula in CNF is a conjunction (AND) of clauses, where each clause is a disjunction (OR) of literals:
```
F = C₁ ∧ C₂ ∧ ... ∧ Cₘ
where each Cᵢ = (l₁ ∨ l₂ ∨ ... ∨ lₖ)
and each lⱼ is either a variable xᵢ or its negation ¬xᵢ
```

## Solution Approaches

### 1. DPLL (Davis-Putnam-Logemann-Loveland)

#### Key Features:
- Complete algorithm (guaranteed to find solution if it exists)
- Recursive backtracking approach
- Uses unit propagation and pure literal elimination

#### Algorithm Steps:
    if all_clauses_satisfied(formula, assignment):
        return True, assignment
    if any_clause_falsified(formula, assignment):
        return False, assignment

    # Unit propagation
    if has_unit_clause(formula):
        literal = get_unit_literal(formula)
        assignment[abs(literal)] = literal > 0
        return dpll(simplify(formula, literal), assignment)

    # Pure literal elimination
    if has_pure_literal(formula):
        literal = get_pure_literal(formula)
        assignment[abs(literal)] = literal > 0
        return dpll(simplify(formula, literal), assignment)

    # Branching
    var = choose_variable(formula)
    # Try True
    assignment[var] = True
    sat, new_assignment = dpll(simplify(formula, var),
                             assignment.copy())
    if sat:
        return True, new_assignment

    # Try False
    assignment[var] = False
    return dpll(simplify(formula, -var), assignment)
```

### 2. Modern SAT Solvers

#### CDCL (Conflict-Driven Clause Learning)
```python
def cdcl_sat(formula):
    """
    Modern SAT solver with conflict learning
    """
    assignment = {}
    learned_clauses = []
    decision_level = 0

    while not all_assigned(formula, assignment):
        conflict = unit_propagate(formula, assignment)
        if conflict:
            if decision_level == 0:
                return False, None  # UNSAT
            backtrack_level = analyze_conflict(conflict)
            learned_clause = derive_learned_clause(conflict)
            formula.add_clause(learned_clause)
            backtrack(backtrack_level)
        else:
            literal = choose_literal(formula, assignment)
            decision_level += 1
            assignment[abs(literal)] = literal > 0

    return True, assignment
```

### 3. Approximation Algorithms

#### Random Walk
```python
def random_walk_sat(formula, max_flips):
    """
    Probabilistic SAT solving
    May not find solution even if one exists
    """
    assignment = random_assignment()

    for _ in range(max_flips):
        if is_satisfied(formula, assignment):
            return True, assignment

        clause = choose_unsatisfied_clause(formula, assignment)
        var = random_variable_from_clause(clause)
        assignment[var] = not assignment[var]

    return False, None
```

## Applications

### 1. Hardware Verification
- Circuit equivalence checking
- Model checking
- Formal verification

### 2. Software Analysis
- Bug finding
- Test generation
- Program verification

### 3. AI Planning
- Action sequence verification
- Goal reachability
- Resource allocation

## Practical Considerations

### 1. Problem Encoding
```python
def encode_graph_coloring_to_sat(graph, colors):
    """
    Example: Reducing graph coloring to SAT
    """
    clauses = []

    # At least one color per vertex
    for v in graph.vertices:
        clauses.append([f"x_{v}_{c}" for c in colors])

    # At most one color per vertex
    for v in graph.vertices:
        for c1 in colors:
            for c2 in colors:
                if c1 < c2:
                    clauses.append([f"-x_{v}_{c1}",
                                  f"-x_{v}_{c2}"])

    # Adjacent vertices have different colors
    for (v1, v2) in graph.edges:
        for c in colors:
            clauses.append([f"-x_{v1}_{c}",
                          f"-x_{v2}_{c}"])

    return clauses
```

### 2. Optimization Techniques
- Variable ordering heuristics
- Clause learning strategies
- Restart policies

### 3. Preprocessing
- Unit propagation
- Pure literal elimination
- Subsumption

## Advanced Topics

### 1. QBF (Quantified Boolean Formulas)
- Extension of SAT with quantifiers
- PSPACE-complete
- Applications in verification

### 2. MaxSAT
- Optimization version of SAT
- Find assignment satisfying maximum clauses
- Applications in optimization

### 3. #SAT
- Counting version of SAT
- Count number of satisfying assignments
- Applications in probabilistic reasoning

## Research Directions

### 1. Algorithm Improvements
- Better learning schemes
- Parallel SAT solving
- Portfolio approaches

### 2. Applications
- Cryptanalysis
- Machine learning
- Constraint solving

### 3. Theoretical Aspects
- Phase transitions
- Proof complexity
- Structure of hard instances

## Best Practices

### 1. Problem Formulation
- Minimize variables and clauses
- Use natural encodings
- Consider symmetry breaking

### 2. Solver Selection
- Problem characteristics
- Performance requirements
- Available resources

### 3. Solution Analysis
- Verify solutions
- Analyze unsatisfiable cores
- Study solving patterns