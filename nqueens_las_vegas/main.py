import random
import matplotlib.pyplot as plt
import numpy as np
from time import time

def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True

def solve_n_queens_backtracking(n=8):
    """Original backtracking algorithm"""
    board = [-1] * n
    row = 0
    state_count = 0
    states_visited = []

    while row >= 0:
        board[row] += 1
        state_count += 1

        # Record current state for visualization
        if row >= 0 and board[row] < n:
            current_state = board.copy()
            # Pad with -1 for incomplete rows
            current_state = current_state[:row+1] + [-1] * (n - row - 1)
            states_visited.append(current_state)

        while board[row] < n and not is_safe(board, row, board[row]):
            board[row] += 1
            state_count += 1

        if board[row] < n:
            if row == n - 1:
                print("Backtracking Solution:", board)
                print("States checked:", state_count)
                return board, state_count, states_visited
            else:
                row += 1
                board[row] = -1
        else:
            board[row] = -1
            row -= 1

    print("No solution found")
    return None, state_count, states_visited

def las_vegas_n_queens(n=8, max_attempts=1000):
    """Las Vegas randomized algorithm for n-queens problem"""
    states_checked = 0
    attempts = 0
    states_visited = []
    start_time = time()

    while attempts < max_attempts:
        attempts += 1

        # Create a board with n queens
        board = [-1] * n
        row = 0
        backtrack = False

        # Place queens row by row with randomized column selection
        while row < n and not backtrack:
            # Create list of available columns for this row
            available_cols = []
            for col in range(n):
                states_checked += 1
                if is_safe(board, row, col):
                    available_cols.append(col)

            # Record current partial state
            current_state = board.copy()
            # Pad with -1 for incomplete rows
            current_state = current_state[:row] + [-1] * (n - row)
            states_visited.append(current_state)

            # If we have available columns, randomly choose one
            if available_cols:
                board[row] = random.choice(available_cols)
                row += 1
            else:
                # No valid position in this row, restart
                backtrack = True

        # If we placed all n queens, we have a solution
        if row == n:
            print("Las Vegas Solution:", board)
            print(f"Found after {attempts} attempts")
            print(f"States checked: {states_checked}")
            print(f"Time taken: {time() - start_time:.4f} seconds")
            return board, states_checked, states_visited

    print("No solution found after", max_attempts, "attempts")
    print(f"States checked: {states_checked}")
    return None, states_checked, states_visited

def min_conflicts_n_queens(n=8, max_steps=1000):
    """Min-conflicts algorithm for n-queens problem"""
    # Start with one queen per row, in random columns
    board = [random.randint(0, n-1) for _ in range(n)]
    states_checked = 0
    states_visited = []
    start_time = time()

    for step in range(max_steps):
        states_visited.append(board.copy())

        # Count conflicts for each queen
        conflicts = []
        for row in range(n):
            conflict_count = 0
            col = board[row]

            for other_row in range(n):
                if other_row == row:
                    continue

                other_col = board[other_row]
                states_checked += 1

                # Check if queens attack each other
                if (other_col == col or
                    other_col - other_row == col - row or
                    other_col + other_row == col + row):
                    conflict_count += 1

            conflicts.append(conflict_count)

        # If no conflicts, we have a solution
        if sum(conflicts) == 0:
            print("Min-Conflicts Solution:", board)
            print(f"Steps taken: {step+1}")
            print(f"States checked: {states_checked}")
            print(f"Time taken: {time() - start_time:.4f} seconds")
            return board, states_checked, states_visited

        # Select a row with conflicts
        conflict_rows = [r for r, c in enumerate(conflicts) if c > 0]
        if not conflict_rows:
            break

        # Pick a random conflicting row
        row = random.choice(conflict_rows)

        # Find the column with minimum conflicts
        min_conflict_cols = []
        min_conflicts = n

        for col in range(n):
            # Count conflicts if queen is placed at this position
            conflict_count = 0
            for other_row in range(n):
                if other_row == row:
                    continue

                other_col = board[other_row]
                states_checked += 1

                # Check if queens attack each other
                if (other_col == col or
                    other_col - other_row == col - row or
                    other_col + other_row == col + row):
                    conflict_count += 1

            # Update min conflicts
            if conflict_count < min_conflicts:
                min_conflicts = conflict_count
                min_conflict_cols = [col]
            elif conflict_count == min_conflicts:
                min_conflict_cols.append(col)

        # Move queen to the position with minimum conflicts (random if multiple)
        board[row] = random.choice(min_conflict_cols)

    print("No solution found after", max_steps, "steps")
    print(f"States checked: {states_checked}")
    return None, states_checked, states_visited

def visualize_board(board, n=8):
    """Visualize a chessboard with queens"""
    board_visual = np.zeros((n, n))

    # Set queen positions
    for i, col in enumerate(board):
        if col != -1:  # Skip unfilled positions
            board_visual[i, col] = 1

    # Create a chessboard pattern for the background
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 1:
                if board_visual[i, j] == 0:  # Don't overwrite queens
                    board_visual[i, j] = 0.2

    plt.figure(figsize=(6, 6))
    plt.imshow(board_visual, cmap='Blues', interpolation='nearest')

    # Add grid
    plt.grid(True, color='black', linewidth=1)
    plt.xticks(np.arange(-.5, n, 1), [])
    plt.yticks(np.arange(-.5, n, 1), [])

    # Add queens
    for i, col in enumerate(board):
        if col != -1:  # Skip unfilled positions
            plt.text(col, i, '♕', fontsize=20, ha='center', va='center')

    plt.title(f"{n}-Queens Solution")

    return plt

def visualize_state_diagram(states, n=8, max_states=10):
    """Visualize state diagram showing progression of algorithm"""
    # Limit number of states shown to avoid overcrowding
    states_to_show = states[:min(len(states), max_states)]
    num_states = len(states_to_show)

    fig, axes = plt.subplots(1, num_states, figsize=(num_states*2, 2))
    if num_states == 1:
        axes = [axes]  # Make it iterable for a single state

    for i, state in enumerate(states_to_show):
        board_visual = np.zeros((n, n))

        # Set queen positions
        for row, col in enumerate(state):
            if col != -1:  # Skip unfilled positions
                board_visual[row, col] = 1

        # Create a chessboard pattern for the background
        for r in range(n):
            for c in range(n):
                if (r + c) % 2 == 1:
                    if board_visual[r, c] == 0:  # Don't overwrite queens
                        board_visual[r, c] = 0.2

        axes[i].imshow(board_visual, cmap='Blues', interpolation='nearest')
        axes[i].set_title(f"State {i+1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Add queens
        for row, col in enumerate(state):
            if col != -1:  # Skip unfilled positions
                axes[i].text(col, row, '♕', fontsize=10, ha='center', va='center')

    plt.tight_layout()
    plt.suptitle(f"First {num_states} States in Algorithm Progress", y=1.1)

    return plt

def compare_algorithms(n=8):
    """Compare all three algorithms"""
    print(f"Solving {n}-Queens Problem with Multiple Algorithms")
    print("-" * 50)

    # Run backtracking
    print("\nRunning Backtracking Algorithm:")
    start_time = time()
    solution1, states1, states_visited1 = solve_n_queens_backtracking(n)
    time1 = time() - start_time

    # Run Las Vegas
    print("\nRunning Las Vegas Algorithm:")
    start_time = time()
    solution2, states2, states_visited2 = las_vegas_n_queens(n)
    time2 = time() - start_time

    # Run Min-Conflicts
    print("\nRunning Min-Conflicts Algorithm:")
    start_time = time()
    solution3, states3, states_visited3 = min_conflicts_n_queens(n)
    time3 = time() - start_time

    # Print comparison
    print("\nAlgorithm Comparison:")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'States Checked':<15} {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Backtracking':<20} {states1:<15} {time1:.6f}")
    print(f"{'Las Vegas':<20} {states2:<15} {time2:.6f}")
    print(f"{'Min-Conflicts':<20} {states3:<15} {time3:.6f}")

    # Visualize solutions
    if solution1:
        plt.figure(1)
        visualize_board(solution1, n)
        plt.savefig('backtracking_solution.png')

        plt.figure(2)
        visualize_state_diagram(states_visited1, n)
        plt.savefig('backtracking_states.png')

    if solution2:
        plt.figure(3)
        visualize_board(solution2, n)
        plt.savefig('las_vegas_solution.png')

        plt.figure(4)
        visualize_state_diagram(states_visited2, n)
        plt.savefig('las_vegas_states.png')

    if solution3:
        plt.figure(5)
        visualize_board(solution3, n)
        plt.savefig('min_conflicts_solution.png')

        plt.figure(6)
        visualize_state_diagram(states_visited3, n)
        plt.savefig('min_conflicts_states.png')

    plt.show()

# Run the comparison
if __name__ == "__main__":
    compare_algorithms(8)

