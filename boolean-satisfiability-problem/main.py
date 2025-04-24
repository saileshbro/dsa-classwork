from typing import List, Tuple, Dict, Optional
import random
from time import time

class ThreeSAT:
    def __init__(self, num_variables: int, num_clauses: int):
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.clauses: List[List[Tuple[str, bool]]] = []

    def generate_random_instance(self) -> None:
        """Generate a random 3-SAT instance"""
        self.clauses = []
        for _ in range(self.num_clauses):
            # Choose 3 distinct variables for this clause
            vars = random.sample(range(1, self.num_variables + 1), 3)
            # Randomly choose whether each variable is negated
            clause = [(f'x{v}', random.choice([True, False])) for v in vars]
            self.clauses.append(clause)

    def calculate_ratio(self) -> float:
        """Calculate the clause-to-variable ratio"""
        return self.num_clauses / self.num_variables

    def estimate_difficulty(self) -> str:
        """Estimate instance difficulty based on ratio"""
        ratio = self.calculate_ratio()
        if ratio < 4.0:
            return "Likely easy (under-constrained)"
        elif 4.0 <= ratio <= 4.5:
            return "Likely hard (phase transition region)"
        else:
            return "May be easier (over-constrained)"

    def evaluate_clause(self, clause: List[Tuple[str, bool]], assignment: Dict[str, bool]) -> bool:
        """Evaluate if a clause is satisfied under the given assignment"""
        for var, is_positive in clause:
            if var in assignment:
                value = assignment[var]
                if is_positive and value:
                    return True
                if not is_positive and not value:
                    return True
        return False

    def is_satisfied(self, assignment: Dict[str, bool]) -> bool:
        """Check if all clauses are satisfied"""
        return all(self.evaluate_clause(clause, assignment) for clause in self.clauses)

    def solve_dpll(self) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """Solve using DPLL algorithm"""
        def dpll_recursive(assignment: Dict[str, bool]) -> Tuple[bool, Optional[Dict[str, bool]]]:
            if len(assignment) == self.num_variables:
                return self.is_satisfied(assignment), assignment

            # Choose an unassigned variable
            for i in range(1, self.num_variables + 1):
                var = f'x{i}'
                if var not in assignment:
                    # Try True first
                    assignment[var] = True
                    solved, result = dpll_recursive(assignment.copy())
                    if solved:
                        return True, result

                    # Try False
                    assignment[var] = False
                    solved, result = dpll_recursive(assignment.copy())
                    if solved:
                        return True, result

                    del assignment[var]
                    break

            return False, None

        return dpll_recursive({})

    def solve_random_walk(self, max_tries: int = 1000, max_flips: int = 1000) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """Solve using random walk algorithm (WalkSAT)"""
        for _ in range(max_tries):
            # Random initial assignment
            assignment = {f'x{i}': random.choice([True, False])
                        for i in range(1, self.num_variables + 1)}

            for _ in range(max_flips):
                if self.is_satisfied(assignment):
                    return True, assignment

                # Find unsatisfied clauses
                unsat_clauses = [clause for clause in self.clauses
                               if not self.evaluate_clause(clause, assignment)]
                if not unsat_clauses:
                    return True, assignment

                # Pick a random unsatisfied clause
                clause = random.choice(unsat_clauses)

                # Flip a random variable in this clause
                var, _ = random.choice(clause)
                assignment[var] = not assignment[var]

        return False, None

def main():
    # Example usage
    num_vars = 20
    num_clauses = 85  # Ratio ≈ 4.27 (phase transition)

    sat_instance = ThreeSAT(num_vars, num_clauses)
    sat_instance.generate_random_instance()

    print(f"3-SAT Instance:")
    print(f"Number of variables: {num_vars}")
    print(f"Number of clauses: {num_clauses}")
    print(f"Clause/Variable ratio: {sat_instance.calculate_ratio():.2f}")
    print(f"Difficulty estimate: {sat_instance.estimate_difficulty()}")
    print("\nClauses:")
    for i, clause in enumerate(sat_instance.clauses, 1):
        clause_str = " ∨ ".join(f"{'¬' if not pos else ''}{var}" for var, pos in clause)
        print(f"{i}. ({clause_str})")

    print("\nSolving using DPLL...")
    start_time = time()
    is_sat, assignment = sat_instance.solve_dpll()
    dpll_time = time() - start_time

    if is_sat:
        print(f"Solution found in {dpll_time:.3f} seconds!")
        print("Satisfying assignment:")
        for var, val in sorted(assignment.items()):
            print(f"{var} = {val}")
    else:
        print("No solution exists (unsatisfiable)")

    print("\nSolving using Random Walk...")
    start_time = time()
    is_sat, assignment = sat_instance.solve_random_walk()
    walk_time = time() - start_time

    if is_sat:
        print(f"Solution found in {walk_time:.3f} seconds!")
        print("Satisfying assignment:")
        for var, val in sorted(assignment.items()):
            print(f"{var} = {val}")
    else:
        print("No solution found (may still be satisfiable)")

if __name__ == "__main__":
    main()
