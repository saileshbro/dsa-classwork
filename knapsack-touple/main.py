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