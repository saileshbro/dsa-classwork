import random
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import pandas as pd

@dataclass
class AlgorithmStats:
    """Statistics for algorithm performance"""
    name: str
    solution: List[int]
    states_checked: int
    time_taken: float
    convergence_path: List[List[int]]
    success_rate: float = 0.0

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

def solve_n_queens_dynamic(n=8):
    """Dynamic Programming approach to N-Queens"""
    def get_first_solution(n, row=0, queens=None, diagonal1=None, diagonal2=None):
        if queens is None:
            queens = set()
        if diagonal1 is None:
            diagonal1 = set()
        if diagonal2 is None:
            diagonal2 = set()

        if row == n:
            # Convert the queens set into a board configuration
            board = [-1] * n
            for r, c in enumerate(sorted(queens)):
                board[r] = c
            return board

        for col in range(n):
            if (col not in queens and
                (row + col) not in diagonal1 and
                (row - col) not in diagonal2):

                queens.add(col)
                diagonal1.add(row + col)
                diagonal2.add(row - col)

                result = get_first_solution(n, row + 1, queens, diagonal1, diagonal2)
                if result:
                    return result

                queens.remove(col)
                diagonal1.remove(row + col)
                diagonal2.remove(row - col)
        return None

    start_time = time()
    board = get_first_solution(n)
    time_taken = time() - start_time

    if board:
        print("Dynamic Programming Solution:", board)
        print(f"Time taken: {time_taken:.4f} seconds")
        return board, n, [board]  # Return single state
    return None, 0, []

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

def enhanced_visualize_board(board, n=8, title="N-Queens Solution"):
    """Enhanced board visualization with better aesthetics"""
    # Changed from seaborn to default style
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    board_visual = np.zeros((n, n))

    # Set queen positions and create enhanced chessboard pattern
    for i, col in enumerate(board):
        if col != -1:
            board_visual[i, col] = 1

    # Create an enhanced chessboard pattern
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 1:
                if board_visual[i, j] == 0:
                    board_visual[i, j] = 0.3

    plt.imshow(board_visual, cmap='RdYlBu_r', interpolation='nearest')

    # Add grid with enhanced styling
    plt.grid(True, color='black', linewidth=1.5, alpha=0.3)
    plt.xticks(np.arange(-.5, n, 1), [])
    plt.yticks(np.arange(-.5, n, 1), [])

    # Add queens with enhanced styling
    for i, col in enumerate(board):
        if col != -1:
            plt.text(col, i, '♕', fontsize=30,
                    ha='center', va='center',
                    color='darkblue')

    plt.title(title, fontsize=16, pad=20)
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

def visualize_algorithm_comparison(stats: List[AlgorithmStats], n: int):
    """Create comprehensive visualization of algorithm comparisons"""
    # Changed from seaborn to default style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # Performance metrics subplot
    plt.subplot(2, 2, 1)
    algorithms = [stat.name for stat in stats]
    times = [stat.time_taken for stat in stats]
    states = [stat.states_checked for stat in stats]

    x = np.arange(len(algorithms))
    width = 0.35

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, times, width, label='Time (s)',
                    color='skyblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, states, width, label='States',
                    color='lightcoral', alpha=0.7)

    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('States Checked', fontsize=12)
    plt.title('Algorithm Performance Comparison', fontsize=14, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Success rate subplot with enhanced styling
    plt.subplot(2, 2, 2)
    success_rates = [stat.success_rate for stat in stats]
    bars = plt.bar(algorithms, success_rates, color='lightgreen', alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Algorithm Success Rates', fontsize=14, pad=20)
    plt.xticks(rotation=45)

    # State space exploration subplot with enhanced styling
    plt.subplot(2, 2, (3, 4))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    for stat, color in zip(stats, colors):
        path_lengths = [len(state) for state in stat.convergence_path[:100]]
        plt.plot(path_lengths, label=stat.name, color=color,
                linewidth=2, alpha=0.7)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Path Length', fontsize=12)
    plt.title('State Space Exploration', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def compare_algorithms(n=8, runs=10):
    """Enhanced comparison of all algorithms including dynamic programming"""
    print(f"Solving {n}-Queens Problem with Multiple Approaches")
    print("-" * 50)

    # Collect performance statistics for all algorithms
    algorithms = [
        (solve_n_queens_backtracking, "Backtracking"),
        (las_vegas_n_queens, "Las Vegas"),
        (min_conflicts_n_queens, "Min-Conflicts"),
        (solve_n_queens_dynamic, "Dynamic Programming")
    ]

    stats = []
    for algo_func, name in algorithms:
        print(f"\nRunning {name} Algorithm:")
        successes = 0
        total_states = 0
        total_time = 0
        solutions = []
        all_paths = []

        for _ in range(runs):
            start_time = time()
            solution, states, path = algo_func(n)
            end_time = time() - start_time

            if solution:
                successes += 1
                solutions.append(solution)
                all_paths.append(path)
            total_states += states
            total_time += end_time

        # Use the first successful solution for visualization
        final_solution = solutions[0] if solutions else None
        final_path = all_paths[0] if all_paths else []

        stat = AlgorithmStats(
            name=name,
            solution=final_solution,
            states_checked=total_states // runs,
            time_taken=total_time / runs,
            convergence_path=final_path,
            success_rate=(successes / runs) * 100
        )
        stats.append(stat)

        # Generate individual algorithm visualizations
        if stat.solution:
            enhanced_visualize_board(
                stat.solution,
                n,
                f"{name} Solution"
            ).savefig(f'{name.lower().replace(" ", "_")}_solution.png')

            visualize_state_diagram(
                stat.convergence_path,
                n
            ).savefig(f'{name.lower().replace(" ", "_")}_states.png')

    # Generate comprehensive comparison visualization
    visualize_algorithm_comparison(stats, n)

    # Print detailed comparison
    print("\nDetailed Algorithm Comparison:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Success Rate':<15} {'Avg States':<15} {'Avg Time (s)':<15}")
    print("-" * 80)

    for stat in stats:
        print(f"{stat.name:<20} {stat.success_rate:>6.2f}% {stat.states_checked:>14} {stat.time_taken:>14.6f}")

    return stats

def plot_comparative_analysis(results_by_size: Dict[int, List[AlgorithmStats]]):
    """Create comparative visualizations across different board sizes"""
    plt.style.use('default')
    board_sizes = sorted(list(results_by_size.keys()))
    algorithms = ["Backtracking", "Las Vegas", "Min-Conflicts", "Dynamic Programming"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Create subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Execution Time Analysis
    ax1 = plt.subplot(2, 2, 1)
    for i, algo in enumerate(algorithms):
        times = [results_by_size[size][i].time_taken * 1000 for size in board_sizes]  # Convert to ms
        plt.plot(board_sizes, times, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Algorithm Execution Time Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 2. States Explored Analysis
    ax2 = plt.subplot(2, 2, 2)
    for i, algo in enumerate(algorithms):
        states = [results_by_size[size][i].states_checked for size in board_sizes]
        plt.plot(board_sizes, states, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('States Explored', fontsize=12)
    plt.title('States Exploration Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 3. Success Rate Analysis
    ax3 = plt.subplot(2, 2, 3)
    for i, algo in enumerate(algorithms):
        success_rates = [results_by_size[size][i].success_rate for size in board_sizes]
        plt.plot(board_sizes, success_rates, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Algorithm Success Rate Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 4. Efficiency Ratio (States/ms)
    ax4 = plt.subplot(2, 2, 4)
    for i, algo in enumerate(algorithms):
        efficiency = []
        for size in board_sizes:
            time_ms = results_by_size[size][i].time_taken * 1000
            if time_ms > 0:
                efficiency.append(results_by_size[size][i].states_checked / time_ms)
            else:
                efficiency.append(0)
        plt.plot(board_sizes, efficiency, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('States Explored per Millisecond', fontsize=12)
    plt.title('Algorithm Efficiency Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_comparative_analysis():
    """Run analysis for different board sizes and generate comparative visualizations"""
    results_by_size = {}
    board_sizes = [8, 12, 16, 20, 24]
    runs = 10

    for size in board_sizes:
        print(f"\nAnalyzing {size}x{size} board...")
        stats = compare_algorithms(size, runs)
        results_by_size[size] = stats

    plot_comparative_analysis(results_by_size)
    print("\nComparative analysis completed. Visualizations saved as 'scaling_analysis.png'")

if __name__ == "__main__":
    run_comparative_analysis()

