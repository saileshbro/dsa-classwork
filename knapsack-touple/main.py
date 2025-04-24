"""
Knapsack Problem State Space Solution Generator
-------------------------------------------
Input:
 o1 o2 o3
value 10 5 20
weight 5 10 5
capacity = 15
"""

from itertools import product
import time

def generate_all_solutions():
    """Generate all possible combinations of 0s and 1s for 3 items"""
    return list(product([0, 1], repeat=3))

def evaluate_solution(tuple_solution, weights, values, capacity):
    """Evaluate a single solution tuple"""
    total_weight = sum(w * x for w, x in zip(weights, tuple_solution))
    total_value = sum(v * x for v, x in zip(values, tuple_solution))
    is_valid = total_weight <= capacity
    return {
        'tuple': tuple_solution,
        'weight': total_weight,
        'value': total_value,
        'valid': is_valid,
        'items_taken': [i+1 for i, x in enumerate(tuple_solution) if x == 1]
    }

def solve_knapsack():
    # Problem parameters
    weights = [5, 10, 5]  # weights of o1, o2, o3
    values = [10, 5, 20]  # values of o1, o2, o3
    capacity = 15

    print("Starting Knapsack Solution Generation...")
    print("-" * 50)
    print(f"Capacity: {capacity}")
    print(f"Weights: {weights}")
    print(f"Values: {values}")
    print("-" * 50)

    # Generate and evaluate all possible solutions
    all_solutions = []
    start_time = time.time()

    print("\nGenerating and evaluating all possible solutions...")
    for solution in generate_all_solutions():
        result = evaluate_solution(solution, weights, values, capacity)
        all_solutions.append(result)

        # Log each solution evaluation
        status = "✓" if result['valid'] else "✗"
        print(f"\nEvaluating solution {solution} {status}")
        print(f"  Items taken: {result['items_taken']}")
        print(f"  Total weight: {result['weight']}/{capacity}")
        print(f"  Total value: {result['value']}")

    # Find the optimal solution
    valid_solutions = [s for s in all_solutions if s['valid']]
    optimal_solution = max(valid_solutions, key=lambda x: x['value'])

    print("\n" + "=" * 50)
    print("Solution Analysis Complete!")
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    print(f"Total solutions generated: {len(all_solutions)}")
    print(f"Valid solutions: {len(valid_solutions)}")
    print("\nOptimal Solution Found:")
    print(f"  Solution tuple: {optimal_solution['tuple']}")
    print(f"  Items to take: {optimal_solution['items_taken']}")
    print(f"  Total weight: {optimal_solution['weight']}/{capacity}")
    print(f"  Total value: {optimal_solution['value']}")
    print("=" * 50)

    return optimal_solution

if __name__ == "__main__":
    # Run the solver
    optimal = solve_knapsack()

    # Verify the solution is correct
    expected_optimal = (1, 0, 1)  # Taking items 1 and 3
    assert optimal['tuple'] == expected_optimal, f"Solution verification failed! Expected {expected_optimal}, got {optimal['tuple']}"
    print("\nSolution verified successfully! ✓")

"""
Tuple (or k-item) Knapsack Problem Implementation
Each item can be taken 0, 1, ..., k times
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

@dataclass
class Item:
    id: int
    value: int
    weight: int
    max_copies: int  # Maximum number of copies allowed

def tuple_knapsack_dp(items: List[Item], capacity: int) -> Tuple[int, Dict[int, int]]:
    """
    Solve tuple knapsack using dynamic programming
    Returns (max_value, item_counts)
    Time: O(n * W * K), Space: O(W)
    where K is maximum copies allowed
    """
    n = len(items)
    dp = [0] * (capacity + 1)
    # Keep track of choices made
    choices = [[0] * n for _ in range(capacity + 1)]

    for w in range(capacity + 1):
        for i, item in enumerate(items):
            for k in range(1, item.max_copies + 1):
                if k * item.weight <= w:
                    val = dp[w - k * item.weight] + k * item.value
                    if val > dp[w]:
                        dp[w] = val
                        choices[w] = choices[w - k * item.weight].copy()
                        choices[w][i] = k

    # Convert choices to dictionary
    item_counts = {item.id: choices[capacity][i]
                  for i, item in enumerate(items)
                  if choices[capacity][i] > 0}

    return dp[capacity], item_counts

def tuple_knapsack_branch_bound(items: List[Item], capacity: int) -> Tuple[int, Dict[int, int]]:
    """
    Branch and bound solution for tuple knapsack
    Returns (max_value, item_counts)
    """
    n = len(items)
    best_value = 0
    best_counts = {}

    def bound(i: int, remaining_capacity: int,
             current_value: int, current_counts: Dict[int, int]) -> float:
        """Calculate upper bound for remaining capacity"""
        if remaining_capacity <= 0:
            return current_value

        bound_value = current_value
        w = remaining_capacity
        j = i

        # Take maximum possible of each remaining item
        while j < n and w > 0:
            item = items[j]
            # Maximum copies possible with remaining capacity
            max_possible = min(item.max_copies,
                             w // item.weight)
            bound_value += max_possible * item.value
            w -= max_possible * item.weight
            j += 1

        return bound_value

    def branch(i: int, remaining_capacity: int,
              current_value: int,
              current_counts: Dict[int, int]):
        nonlocal best_value, best_counts

        if i >= n or remaining_capacity <= 0:
            if current_value > best_value:
                best_value = current_value
                best_counts = current_counts.copy()
            return

        # Skip if bound is worse than current best
        if bound(i, remaining_capacity,
                current_value, current_counts) <= best_value:
            return

        item = items[i]
        # Try different numbers of copies
        for copies in range(item.max_copies + 1):
            if copies * item.weight <= remaining_capacity:
                if copies > 0:
                    current_counts[item.id] = copies
                branch(i + 1,
                      remaining_capacity - copies * item.weight,
                      current_value + copies * item.value,
                      current_counts)
                if copies > 0:
                    del current_counts[item.id]

    branch(0, capacity, 0, {})
    return best_value, best_counts

def tuple_knapsack_fptas(items: List[Item], capacity: int,
                        epsilon: float) -> Tuple[int, Dict[int, int]]:
    """
    FPTAS for tuple knapsack
    Guarantees (1-ε) optimal solution
    Running time: O(n²/ε)
    """
    # Scale values
    max_value = max(item.value for item in items)
    K = (epsilon * max_value) / len(items)

    scaled_items = [
        Item(
            id=item.id,
            value=int(item.value / K),
            weight=item.weight,
            max_copies=item.max_copies
        )
        for item in items
    ]

    # Solve with scaled values
    _, counts = tuple_knapsack_dp(scaled_items, capacity)

    # Calculate actual value
    real_value = sum(
        items[i].value * count
        for i, (item_id, count) in enumerate(counts.items())
    )

    return real_value, counts

def genetic_tuple_knapsack(items: List[Item], capacity: int,
                          pop_size: int = 100,
                          generations: int = 1000,
                          mutation_rate: float = 0.1) -> Tuple[int, Dict[int, int]]:
    """
    Genetic algorithm for tuple knapsack
    Returns (max_value, item_counts)
    """
    def create_individual():
        """Create random valid solution"""
        counts = {}
        remaining = capacity

        for item in items:
            max_possible = min(item.max_copies,
                             remaining // item.weight)
            if max_possible > 0:
                copies = random.randint(0, max_possible)
                if copies > 0:
                    counts[item.id] = copies
                    remaining -= copies * item.weight
        return counts

    def fitness(counts: Dict[int, int]) -> int:
        """Calculate value of solution"""
        total_weight = sum(
            items[i].weight * count
            for i, (item_id, count) in enumerate(counts.items())
        )
        if total_weight > capacity:
            return 0
        return sum(
            items[i].value * count
            for i, (item_id, count) in enumerate(counts.items())
        )

    def crossover(parent1: Dict[int, int],
                 parent2: Dict[int, int]) -> Dict[int, int]:
        """Create child solution from parents"""
        child = {}
        remaining = capacity

        # Randomly choose counts from either parent
        for item in items:
            if random.random() < 0.5:
                count = parent1.get(item.id, 0)
            else:
                count = parent2.get(item.id, 0)

            if count > 0 and count * item.weight <= remaining:
                child[item.id] = count
                remaining -= count * item.weight

        return child

    def mutate(solution: Dict[int, int]):
        """Randomly modify solution"""
        if random.random() > mutation_rate:
            return

        # Either add or remove copies of random item
        item = random.choice(items)
        current = solution.get(item.id, 0)

        if current > 0 and random.random() < 0.5:
            # Remove copies
            new_count = random.randint(0, current - 1)
            if new_count == 0:
                del solution[item.id]
            else:
                solution[item.id] = new_count
        else:
            # Try to add copies
            remaining = capacity - sum(
                items[i].weight * count
                for i, (item_id, count) in enumerate(solution.items())
            )
            max_possible = min(item.max_copies,
                             remaining // item.weight)
            if max_possible > current:
                solution[item.id] = random.randint(
                    current + 1, max_possible
                )

    # Initialize population
    population = [create_individual() for _ in range(pop_size)]
    best_solution = None
    best_fitness = 0

    for _ in range(generations):
        # Evaluate fitness
        population = sorted(
            population,
            key=fitness,
            reverse=True
        )

        # Update best solution
        if fitness(population[0]) > best_fitness:
            best_solution = population[0].copy()
            best_fitness = fitness(population[0])

        # Create new generation
        new_pop = population[:pop_size//2]  # Keep best half

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:pop_size//2], 2)
            child = crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        population = new_pop

    return best_fitness, best_solution

def compare_algorithms(items: List[Item], capacity: int):
    """Compare different algorithms for tuple knapsack"""

    print("Dynamic Programming Solution:")
    dp_value, dp_counts = tuple_knapsack_dp(items, capacity)
    print(f"Value: {dp_value}")
    print(f"Items: {dp_counts}")

    print("\nBranch and Bound Solution:")
    bb_value, bb_counts = tuple_knapsack_branch_bound(
        items, capacity
    )
    print(f"Value: {bb_value}")
    print(f"Items: {bb_counts}")

    print("\nFPTAS Solution (ε=0.1):")
    fptas_value, fptas_counts = tuple_knapsack_fptas(
        items, capacity, 0.1
    )
    print(f"Value: {fptas_value}")
    print(f"Items: {fptas_counts}")

    print("\nGenetic Algorithm Solution:")
    ga_value, ga_counts = genetic_tuple_knapsack(
        items, capacity
    )
    print(f"Value: {ga_value}")
    print(f"Items: {ga_counts}")

if __name__ == "__main__":
    # Example usage
    test_items = [
        Item(0, 6, 2, 3),  # value=6, weight=2, max=3 copies
        Item(1, 10, 4, 2), # value=10, weight=4, max=2 copies
        Item(2, 12, 6, 2), # value=12, weight=6, max=2 copies
        Item(3, 15, 9, 1)  # value=15, weight=9, max=1 copy
    ]

    test_capacity = 15
    compare_algorithms(test_items, test_capacity)

    # Additional analysis could be added here
    print("\nAlgorithm Comparison:")
    print("1. Dynamic Programming:")
    print("   - Exact solution")
    print("   - O(nWK) time, O(W) space")
    print("2. Branch and Bound:")
    print("   - Exact solution")
    print("   - Better for sparse solutions")
    print("3. FPTAS:")
    print("   - Approximation scheme")
    print("   - Guaranteed (1-ε) optimal")
    print("4. Genetic Algorithm:")
    print("   - Heuristic approach")
    print("   - Good for large instances")