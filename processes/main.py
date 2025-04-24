"""
Process Scheduling Algorithms Implementation
Demonstrates deterministic and non-deterministic approaches
"""

import random
from dataclasses import dataclass
from typing import List, Optional
from queue import PriorityQueue
import statistics

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: Optional[int] = None
    remaining_time: Optional[int] = None
    completion_time: Optional[int] = None

    def __post_init__(self):
        if self.remaining_time is None:
            self.remaining_time = self.burst_time

class Scheduler:
    def __init__(self):
        self.current_time = 0
        self.processes: List[Process] = []
        self.completed_processes: List[Process] = []

    def add_process(self, process: Process):
        self.processes.append(process)

    def reset(self):
        self.current_time = 0
        for process in self.processes:
            process.remaining_time = process.burst_time
            process.completion_time = None
        self.completed_processes = []

    def is_complete(self) -> bool:
        return len(self.completed_processes) == len(self.processes)

    def get_metrics(self):
        if not self.completed_processes:
            return None

        turnaround_times = []
        waiting_times = []

        for process in self.completed_processes:
            turnaround = process.completion_time - process.arrival_time
            waiting = turnaround - process.burst_time
            turnaround_times.append(turnaround)
            waiting_times.append(waiting)

        return {
            'avg_turnaround': statistics.mean(turnaround_times),
            'avg_waiting': statistics.mean(waiting_times),
            'max_completion': max(p.completion_time
                                for p in self.completed_processes)
        }

class FCFSScheduler(Scheduler):
    """First Come First Serve - Deterministic"""

    def run(self):
        self.reset()
        ready_queue = sorted(self.processes,
                           key=lambda p: p.arrival_time)

        while ready_queue:
            process = ready_queue[0]

            if process.arrival_time > self.current_time:
                self.current_time = process.arrival_time

            # Execute process
            self.current_time += process.burst_time
            process.completion_time = self.current_time

            self.completed_processes.append(process)
            ready_queue.pop(0)

class SJFScheduler(Scheduler):
    """Shortest Job First - Deterministic"""

    def run(self):
        self.reset()
        remaining = self.processes.copy()
        ready_queue = []

        while remaining or ready_queue:
            # Add newly arrived processes to ready queue
            while remaining and \
                  remaining[0].arrival_time <= self.current_time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                self.current_time = remaining[0].arrival_time
                continue

            # Select shortest job
            process = min(ready_queue,
                        key=lambda p: p.burst_time)
            ready_queue.remove(process)

            # Execute process
            self.current_time += process.burst_time
            process.completion_time = self.current_time

            self.completed_processes.append(process)

class RoundRobinScheduler(Scheduler):
    """Round Robin - Deterministic"""

    def __init__(self, quantum: int = 2):
        super().__init__()
        self.quantum = quantum

    def run(self):
        self.reset()
        remaining = self.processes.copy()
        ready_queue = []

        while remaining or ready_queue:
            # Add newly arrived processes
            while remaining and \
                  remaining[0].arrival_time <= self.current_time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                self.current_time = remaining[0].arrival_time
                continue

            # Execute process for quantum time
            process = ready_queue.pop(0)
            execution_time = min(self.quantum,
                               process.remaining_time)

            self.current_time += execution_time
            process.remaining_time -= execution_time

            # Either complete or requeue process
            if process.remaining_time == 0:
                process.completion_time = self.current_time
                self.completed_processes.append(process)
            else:
                ready_queue.append(process)

class RandomScheduler(Scheduler):
    """Random Scheduling - Non-deterministic"""

    def run(self):
        self.reset()
        remaining = self.processes.copy()
        ready_queue = []

        while remaining or ready_queue:
            # Add newly arrived processes
            while remaining and \
                  remaining[0].arrival_time <= self.current_time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                self.current_time = remaining[0].arrival_time
                continue

            # Randomly select process
            process = random.choice(ready_queue)
            ready_queue.remove(process)

            # Execute process
            self.current_time += process.burst_time
            process.completion_time = self.current_time

            self.completed_processes.append(process)

class LotteryScheduler(Scheduler):
    """Lottery Scheduling - Non-deterministic"""

    def run(self, time_slice: int = 1):
        self.reset()
        remaining = self.processes.copy()
        ready_queue = []

        while remaining or ready_queue:
            # Add newly arrived processes
            while remaining and \
                  remaining[0].arrival_time <= self.current_time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                self.current_time = remaining[0].arrival_time
                continue

            # Assign tickets based on priority
            total_tickets = sum(p.priority or 1
                              for p in ready_queue)
            winning_ticket = random.randrange(total_tickets)

            # Select winning process
            ticket_count = 0
            for process in ready_queue:
                ticket_count += process.priority or 1
                if ticket_count > winning_ticket:
                    # Execute process for time slice
                    self.current_time += min(time_slice,
                                           process.remaining_time)
                    process.remaining_time -= time_slice

                    if process.remaining_time <= 0:
                        process.completion_time = self.current_time
                        self.completed_processes.append(process)
                        ready_queue.remove(process)
                    break

def compare_schedulers(processes: List[Process],
                      num_runs: int = 100):
    """Compare different scheduling algorithms"""
    schedulers = {
        'FCFS': FCFSScheduler(),
        'SJF': SJFScheduler(),
        'RR': RoundRobinScheduler(quantum=2),
        'Random': RandomScheduler(),
        'Lottery': LotteryScheduler()
    }

    results = {}

    for name, scheduler in schedulers.items():
        print(f"\nTesting {name} Scheduler:")

        # Add processes to scheduler
        for process in processes:
            scheduler.add_process(process)

        # Run multiple times for non-deterministic
        metrics_list = []
        for _ in range(num_runs if 'Random' in name
                      or 'Lottery' in name else 1):
            scheduler.run()
            metrics = scheduler.get_metrics()
            if metrics:
                metrics_list.append(metrics)

        # Calculate averages
        if metrics_list:
            avg_metrics = {
                'avg_turnaround': statistics.mean(
                    m['avg_turnaround'] for m in metrics_list
                ),
                'avg_waiting': statistics.mean(
                    m['avg_waiting'] for m in metrics_list
                ),
                'avg_completion': statistics.mean(
                    m['max_completion'] for m in metrics_list
                )
            }
            results[name] = avg_metrics
            print(f"Average Metrics over {num_runs} runs:")
            print(f"Avg Turnaround Time: {avg_metrics['avg_turnaround']:.2f}")
            print(f"Avg Waiting Time: {avg_metrics['avg_waiting']:.2f}")
            print(f"Avg Completion Time: {avg_metrics['avg_completion']:.2f}")

    return results

if __name__ == "__main__":
    # Example usage
    test_processes = [
        Process(1, 0, 6, priority=2),
        Process(2, 2, 4, priority=1),
        Process(3, 4, 8, priority=3),
        Process(4, 6, 3, priority=2),
        Process(5, 8, 5, priority=1)
    ]

    results = compare_schedulers(test_processes)

    # Additional analysis could be added here
    print("\nPerformance Comparison:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Normalized Performance Score: "
              f"{1/metrics['avg_waiting']:.3f}")
