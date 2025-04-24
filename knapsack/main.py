"""
Knapsack Problem Implementation

Items:
- Value Weight
1 60 10
2 100 20
3 120 30
"""

def knapsack_dynamic_programming(values, weights, capacity):
    """
    Solve the knapsack problem using dynamic programming approach.

    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack

    Returns:
        Maximum value that can be put in the knapsack
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

    # Return the maximum value
    return dp[n][capacity]

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

# Example from the problem data
if __name__ == "__main__":
    # Example data from the comments
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    # Find maximum value possible
    max_value = knapsack_dynamic_programming(values, weights, capacity)

    # Find items included in the optimal solution
    items_included = knapsack_items_included(values, weights, capacity)

    print(f"Maximum value in knapsack: {max_value}")
    print(f"Items included (0-indexed): {items_included}")
    print("Items included (1-indexed):", [i+1 for i in items_included])
