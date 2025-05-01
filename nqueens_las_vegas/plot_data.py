import matplotlib.pyplot as plt
import numpy as np

# Algorithm performance data
board_sizes = [8, 12, 16, 20, 24]

# States checked data
states_data = {
    'Backtracking': [981, 3315, 170748, 4192125, 10289900],
    'Las Vegas': [596, 1999, 5382, 20342, 14592],
    'Min-Conflicts': [4928, 39666, 63456, 78812, 98035],
    'Dynamic Programming': [8, 12, 16, 20, 24]
}

# Time taken data (in seconds)
time_data = {
    'Backtracking': [0.000350, 0.001406, 0.093227, 2.610903, 7.161032],
    'Las Vegas': [0.000195, 0.000708, 0.002232, 0.011258, 0.011405],
    'Min-Conflicts': [0.000603, 0.004419, 0.006906, 0.008457, 0.010317],
    'Dynamic Programming': [0.000082, 0.000229, 0.010123, 0.230590, 0.510515]
}

# Success rate data (in percentage)
success_rate_data = {
    'Backtracking': [100, 100, 100, 100, 100],
    'Las Vegas': [100, 100, 100, 100, 100],
    'Min-Conflicts': [100, 90, 100, 100, 100],
    'Dynamic Programming': [100, 100, 100, 100, 100]
}

def plot_states_comparison():
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (algo, states) in enumerate(states_data.items()):
        plt.plot(board_sizes, states, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('States Explored', fontsize=12)
    plt.title('States Exploration Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('states_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_comparison():
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (algo, times) in enumerate(time_data.items()):
        plt.plot(board_sizes, times, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Algorithm Execution Time Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate_comparison():
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (algo, rates) in enumerate(success_rate_data.items()):
        plt.plot(board_sizes, rates, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Algorithm Success Rate Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(85, 101)  # Set y-axis limits to better show the differences
    plt.tight_layout()
    plt.savefig('success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_comparison():
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, algo in enumerate(states_data.keys()):
        efficiency = []
        for states, time in zip(states_data[algo], time_data[algo]):
            if time > 0:
                efficiency.append(states / time)
            else:
                efficiency.append(0)
        plt.plot(board_sizes, efficiency, marker='o', label=algo, color=colors[i], linewidth=2)

    plt.xlabel('Board Size (N×N)', fontsize=12)
    plt.ylabel('States Explored per Second', fontsize=12)
    plt.title('Algorithm Efficiency Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_states_comparison()
    plot_time_comparison()
    plot_success_rate_comparison()
    plot_efficiency_comparison()