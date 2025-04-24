"""
Boolean Satisfiability Problem (SAT) with tree pruning
Example with multiple clauses to demonstrate pruning
"""

from typing import List, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, assignment: Dict[int, bool], level: int):
        self.assignment = assignment
        self.level = level
        self.pruned = False

    def __str__(self):
        if not self.assignment:
            return "Root"
        return " ".join(f"x{k}={'1' if v else '0'}" for k, v in sorted(self.assignment.items()))

class SATSolver:
    def __init__(self, num_variables: int, clauses: List[List[tuple]]):
        self.num_variables = num_variables
        self.clauses = clauses
        self.tree_output = []
        self.solution_found = False
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def evaluate_clause(self, clause: List[tuple], assignment: Dict[int, bool]) -> bool:
        for var, is_positive in clause:
            if var in assignment:
                if is_positive and assignment[var]:
                    return True
                if not is_positive and not assignment[var]:
                    return True
        return False

    def is_satisfiable(self, assignment: Dict[int, bool]) -> bool:
        return all(self.evaluate_clause(clause, assignment) for clause in self.clauses)

    def can_be_satisfied(self, assignment: Dict[int, bool]) -> bool:
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

    def print_tree_node(self, node: Node, prefix: str, is_last: bool, pruned: bool = False, reason: str = ""):
        """Print a single node of the tree"""
        marker = "└── " if is_last else "├── "
        status = ""
        if pruned:
            status = f" (pruned: {reason})"
        elif self.solution_found and len(node.assignment) == self.num_variables:
            status = " (solution!)"
        elif len(node.assignment) == self.num_variables:
            status = " (invalid)"

        self.tree_output.append(f"{prefix}{marker}{str(node)}{status}")
        return prefix + ("    " if is_last else "│   ")

    def solve(self) -> Optional[Dict[int, bool]]:
        """Solve the SAT problem with tree pruning and visualization"""
        self.tree_output = []
        self.solution_found = False
        self.nodes_explored = 0
        self.nodes_pruned = 0

        def backtrack(assignment: Dict[int, bool], level: int, prefix: str = "", is_last: bool = True) -> Optional[Dict[int, bool]]:
            self.nodes_explored += 1
            current_node = Node(assignment.copy(), level)

            # Check if we can prune this branch
            if not self.can_be_satisfied(assignment):
                self.nodes_pruned += 1
                clause_num = next(i for i, clause in enumerate(self.clauses, 1)
                                if not any((var not in assignment or
                                        (is_positive and assignment[var]) or
                                        (not is_positive and not assignment[var]))
                                        for var, is_positive in clause))
                self.print_tree_node(current_node, prefix, is_last, pruned=True,
                                   reason=f"clause {clause_num} unsatisfiable")
                return None

            # If we have a complete assignment, check if it's a solution
            if len(assignment) == self.num_variables:
                self.print_tree_node(current_node, prefix, is_last)
                if self.is_satisfiable(assignment):
                    self.solution_found = True
                    return assignment
                return None

            var = level + 1
            new_prefix = self.print_tree_node(current_node, prefix, is_last)

            # Try True (1) - left branch
            assignment[var] = True
            result = backtrack(assignment.copy(), level + 1, new_prefix, False)
            if result:
                return result

            # Try False (0) - right branch
            assignment[var] = False
            result = backtrack(assignment.copy(), level + 1, new_prefix, True)
            if result:
                return result

            return None

        solution = backtrack({}, 0)

        # Print the tree visualization with statistics
        print("\nState Space Tree (with pruning):")
        print("--------------------------------")
        for line in self.tree_output:
            print(line)
        print("--------------------------------")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Nodes pruned: {self.nodes_pruned}")
        print(f"Efficiency: {(self.nodes_pruned / self.nodes_explored * 100):.1f}% of nodes pruned")

        return solution

def main():
    # Example with contradictory clauses to show pruning:
    # (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2) ∧ (¬x2 ∨ x3) ∧ (¬x2 ∨ ¬x3)
    print("SAT Formula:")
    print("(x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2) ∧ (¬x2 ∨ x3) ∧ (¬x2 ∨ ¬x3)")
    print()

    num_variables = 3
    clauses = [
        [(1, True), (2, True)],      # x1 v x2
        [(1, False), (2, True)],     # ~x1 v x2
        [(1, True), (2, False)],     # x1 v ~x2
        [(2, False), (3, True)],     # ~x2 v x3
        [(2, False), (3, False)]     # ~x2 v ~x3
    ]

    solver = SATSolver(num_variables, clauses)
    solution = solver.solve()

    if solution:
        print("\nSAT Problem is satisfiable!")
        print("Solution:", {f"x{k}": v for k, v in solution.items()})
    else:
        print("\nSAT Problem is unsatisfiable!")

if __name__ == "__main__":
    main()
