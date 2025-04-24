"""
Knapsack Problem Implementation with Visualization

Items:
- Value Weight
1 60 10
2 100 20
3 120 30
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import networkx as nx

def knapsack_dynamic_programming(values, weights, capacity):
    """
    Solve the knapsack problem using dynamic programming approach.

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack

    Returns:
        Maximum value that can be put in the knapsack and the DP table
    """
    n = len(values)
    # Create a 2D array for memoization
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build table dp[][] in bottom up manner
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    # Visualize the DP table as a tree
    visualize_dp_table(dp, values, weights)

    # Return the maximum value
    return dp[n][capacity], dp

def knapsack_items_included(values, weights, capacity):
    """
    Determine which items are included in the optimal solution.

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack

    Returns:
        List of indices of items included in the knapsack
    """
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build the dp table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    # Find the items included
    w = capacity
    included_items = []

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            included_items.append(i-1)
            w -= weights[i-1]

    return sorted(included_items)

def visualize_knapsack_solution(values: List[int], weights: List[int], capacity: int, included_items: List[int]) -> None:
    """
    Visualize the knapsack problem solution using matplotlib.

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack
        included_items: List of indices of items included in the solution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Items comparison
    x = np.arange(len(values))
    width = 0.35

    ax1.bar(x - width/2, values, width, label='Value')
    ax1.bar(x + width/2, weights, width, label='Weight')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Item {i+1}' for i in range(len(values))])
    ax1.set_ylabel('Value/Weight')
    ax1.set_title('Items Values and Weights')
    ax1.legend()

    # Highlight selected items
    for idx in included_items:
        ax1.axvspan(idx-0.4, idx+0.4, alpha=0.2, color='green')

    # Plot 2: Solution visualization
    total_value = sum(values[i] for i in included_items)
    total_weight = sum(weights[i] for i in included_items)

    ax2.pie([total_weight, capacity - total_weight],
            labels=[f'Used\n({total_weight})', f'Remaining\n({capacity - total_weight})'],
            colors=['lightcoral', 'lightgreen'],
            autopct='%1.1f%%')
    ax2.set_title(f'Capacity Usage\nTotal Value: {total_value}')

    plt.tight_layout()
    plt.savefig('knapsack_solution.png')
    plt.close()

def print_solution_details(values: List[int], weights: List[int], capacity: int, included_items: List[int]) -> None:
    """
    Print detailed solution information.

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack
        included_items: List of indices of items included in the solution
    """
    print("\n===== Knapsack Problem Solution =====")
    print(f"\nKnapsack Capacity: {capacity}")

    print("\nAvailable Items:")
    print("Item\tValue\tWeight\tValue/Weight")
    print("-" * 35)
    for i in range(len(values)):
        print(f"{i+1}\t{values[i]}\t{weights[i]}\t{values[i]/weights[i]:.2f}")

    print("\nSelected Items:")
    print("Item\tValue\tWeight")
    print("-" * 25)
    total_value = 0
    total_weight = 0
    for idx in included_items:
        print(f"{idx+1}\t{values[idx]}\t{weights[idx]}")
        total_value += values[idx]
        total_weight += weights[idx]

    print("\nSummary:")
    print(f"Total Value: {total_value}")
    print(f"Total Weight: {total_weight}/{capacity}")
    print(f"Capacity Utilization: {(total_weight/capacity)*100:.2f}%")
    print("\nVisualization saved as 'knapsack_solution.png'")

def knapsack_recursive(values: List[int], weights: List[int], capacity: int, n: int, calls=None, parent=None) -> Tuple[int, dict]:
    """
    Solve knapsack problem using recursive approach (for visualization).

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity
        n: Number of items
        calls: Dictionary to track recursive calls for visualization
        parent: Parent node ID for visualization

    Returns:
        Tuple of (maximum value, call tree dictionary)
    """
    if calls is None:
        calls = {'counter': 0, 'nodes': {}, 'edges': set()}

    current_id = calls['counter']
    calls['counter'] += 1

    # Add node information
    node_label = f"n={n},cap={capacity}"
    calls['nodes'][current_id] = node_label

    # Add edge to parent if exists
    if parent is not None:
        calls['edges'].add((parent, current_id))

    if n == 0 or capacity == 0:
        return 0, calls

    if weights[n-1] > capacity:
        result, calls = knapsack_recursive(values, weights, capacity, n-1, calls, current_id)
        return result, calls

    # Include item
    include_val, calls = knapsack_recursive(values, weights, capacity-weights[n-1], n-1, calls, current_id)
    include_val += values[n-1]

    # Exclude item
    exclude_val, calls = knapsack_recursive(values, weights, capacity, n-1, calls, current_id)

    return max(include_val, exclude_val), calls

def visualize_recursive_tree(calls: dict, filename: str = 'knapsack_recursive_tree.png'):
    """
    Visualize the recursive call tree.

    Args:
        calls: Dictionary containing call tree information
        filename: Output file name
    """
    G = nx.DiGraph()

    # Add nodes
    for node_id, label in calls['nodes'].items():
        G.add_node(node_id, label=label)

    # Add edges
    for edge in calls['edges']:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=50)

    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color='lightblue', node_size=2000, arrowsize=20,
            font_size=8, font_weight='bold')

    plt.title('Recursive Solution Tree')
    plt.savefig(filename)
    plt.close()

def visualize_dp_table(dp: List[List[int]], values: List[int], weights: List[int],
                      filename: str = 'knapsack_dp_tree.png'):
    """
    Visualize the dynamic programming solution table as a tree.

    Args:
        dp: Dynamic programming table
        values: List of values for each item
        weights: List of weights for each item
        filename: Output file name
    """
    G = nx.DiGraph()
    node_counter = 0

    # Create nodes for each cell in the dp table
    nodes = {}
    for i in range(len(dp)):
        for w in range(len(dp[0])):
            node_id = node_counter
            label = f"i={i},w={w}\nv={dp[i][w]}"
            G.add_node(node_id, label=label)
            nodes[(i, w)] = node_id
            node_counter += 1

    # Add edges showing dependencies
    for i in range(1, len(dp)):
        for w in range(len(dp[0])):
            current = nodes[(i, w)]
            # Edge to cell above (exclude item case)
            G.add_edge(nodes[(i-1, w)], current)

            # Edge to cell above and left (include item case)
            if w >= weights[i-1]:
                G.add_edge(nodes[(i-1, w-weights[i-1])], current)

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=50)

    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color='lightgreen', node_size=2000, arrowsize=20,
            font_size=8, font_weight='bold')

    plt.title('Dynamic Programming Solution Tree')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Example data from the comments
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    # Solve using recursive approach and visualize the call tree
    max_value_recursive, calls = knapsack_recursive(values, weights, capacity, len(values))
    visualize_recursive_tree(calls)

    # Solve using dynamic programming and visualize the solution
    max_value_dp, dp_table = knapsack_dynamic_programming(values, weights, capacity)

    # Find items included in the optimal solution
    items_included = knapsack_items_included(values, weights, capacity)

    # Visualize the solution
    visualize_knapsack_solution(values, weights, capacity, items_included)

    # Print detailed solution
    print("\n===== Knapsack Problem Solution =====")
    print("\nRecursive Approach:")
    print(f"Maximum Value: {max_value_recursive}")
    print("Tree visualization saved as 'knapsack_recursive_tree.png'")

    print("\nDynamic Programming Approach:")
    print(f"Maximum Value: {max_value_dp}")
    print("Tree visualization saved as 'knapsack_dp_tree.png'")

    print_solution_details(values, weights, capacity, items_included)
