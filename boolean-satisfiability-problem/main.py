"""
Boolean Satisfiability Problem (SAT) Solver Implementation
Including DPLL, CDCL, and local search approaches
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random
import time

@dataclass
class Literal:
    variable: int
    positive: bool

    def __neg__(self):
        return Literal(self.variable, not self.positive)

    def __str__(self):
        return f"{'' if self.positive else '¬'}x{self.variable}"

@dataclass
class Clause:
    literals: List[Literal]

    def __str__(self):
        return "(" + " ∨ ".join(str(lit) for lit in self.literals) + ")"

class CNFFormula:
    def __init__(self, clauses: List[Clause]):
        self.clauses = clauses
        self.num_vars = max(
            lit.variable for clause in clauses
            for lit in clause.literals
        )

    def evaluate(self, assignment: Dict[int, bool]) -> bool:
        """Evaluate formula under given assignment"""
        return all(
            any(lit.positive == assignment.get(lit.variable, False)
                for lit in clause.literals)
            for clause in self.clauses
        )

    def __str__(self):
        return " ∧ ".join(str(clause) for clause in self.clauses)

class DPLLSolver:
    """
    Davis-Putnam-Logemann-Loveland (DPLL) Algorithm
    Complete SAT solver using backtracking
    """
    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.assignment: Dict[int, bool] = {}

    def solve(self) -> Optional[Dict[int, bool]]:
        """Return satisfying assignment or None if UNSAT"""
        return self._dpll(self.formula.clauses.copy())

    def _dpll(self,
             clauses: List[Clause]) -> Optional[Dict[int, bool]]:
        """Recursive DPLL procedure"""
        # Unit propagation
        while True:
            unit_clauses = [c for c in clauses if len(c.literals) == 1]
            if not unit_clauses:
                break

            # Propagate unit literal
            lit = unit_clauses[0].literals[0]
            self.assignment[lit.variable] = lit.positive

            # Simplify formula
            clauses = self._simplify(clauses, lit)

            if not clauses:  # All clauses satisfied
                return self.assignment
            if any(not c.literals for c in clauses):  # Empty clause
                return None

        # Pure literal elimination
        pure_lits = self._find_pure_literals(clauses)
        if pure_lits:
            for lit in pure_lits:
                self.assignment[lit.variable] = lit.positive
                clauses = self._simplify(clauses, lit)

            if not clauses:  # All clauses satisfied
                return self.assignment

        # Choose variable and branch
        if not clauses:
            return self.assignment

        var = self._choose_variable(clauses)

        # Try True
        self.assignment[var] = True
        new_clauses = self._simplify(
            clauses,
            Literal(var, True)
        )
        result = self._dpll(new_clauses)
        if result is not None:
            return result

        # Try False
        self.assignment[var] = False
        new_clauses = self._simplify(
            clauses,
            Literal(var, False)
        )
        return self._dpll(new_clauses)

    def _simplify(self,
                 clauses: List[Clause],
                 lit: Literal) -> List[Clause]:
        """Simplify formula by assigning literal"""
        result = []
        for clause in clauses:
            # Skip satisfied clauses
            if any(l.variable == lit.variable and
                   l.positive == lit.positive
                   for l in clause.literals):
                continue

            # Remove falsified literals
            new_lits = [l for l in clause.literals
                       if l.variable != lit.variable]
            if new_lits:
                result.append(Clause(new_lits))

        return result

    def _find_pure_literals(self,
                          clauses: List[Clause]) -> List[Literal]:
        """Find pure literals in formula"""
        polarities = defaultdict(set)
        for clause in clauses:
            for lit in clause.literals:
                polarities[lit.variable].add(lit.positive)

        return [Literal(var, next(iter(pols)))
                for var, pols in polarities.items()
                if len(pols) == 1]

    def _choose_variable(self,
                       clauses: List[Clause]) -> int:
        """Choose next variable for branching"""
        # Use VSIDS (Variable State Independent Decaying Sum)
        scores = defaultdict(int)
        for clause in clauses:
            for lit in clause.literals:
                scores[lit.variable] += 1

        return max(scores.keys(),
                  key=lambda v: scores[v])

class CDCLSolver:
    """
    Conflict-Driven Clause Learning (CDCL)
    Modern complete SAT solver
    """
    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.assignment: Dict[int, bool] = {}
        self.level: Dict[int, int] = {}  # Decision level
        self.antecedent: Dict[int, Clause] = {}  # Reason for propagation
        self.learned_clauses: List[Clause] = []

    def solve(self) -> Optional[Dict[int, bool]]:
        """Return satisfying assignment or None if UNSAT"""
        while True:
            conflict = self._unit_propagate()
            if conflict is None:
                if len(self.assignment) == self.formula.num_vars:
                    return self.assignment

                # Make new decision
                var = self._choose_variable()
                self._assign(var, True, None, len(self.level))

            else:  # Conflict occurred
                level = self._analyze_conflict(conflict)
                if level < 0:  # UNSAT
                    return None
                self._backtrack(level)

    def _unit_propagate(self) -> Optional[Clause]:
        """Perform unit propagation, return conflict if any"""
        while True:
            propagated = False
            for clause in self.formula.clauses + self.learned_clauses:
                # Count unassigned and satisfied literals
                unassigned = []
                sat = False
                for lit in clause.literals:
                    if lit.variable not in self.assignment:
                        unassigned.append(lit)
                    elif self.assignment[lit.variable] == lit.positive:
                        sat = True
                        break

                if sat:
                    continue

                if not unassigned:  # Conflict
                    return clause

                if len(unassigned) == 1:  # Unit clause
                    lit = unassigned[0]
                    self._assign(lit.variable,
                               lit.positive,
                               clause,
                               max(self._clause_level(clause), 0))
                    propagated = True

            if not propagated:
                return None

    def _analyze_conflict(self,
                        conflict: Clause) -> int:
        """
        Analyze conflict and learn new clause
        Returns backtrack level
        """
        curr_level = len(self.level)
        if curr_level == 0:
            return -1  # UNSAT

        # Compute First UIP (Unique Implication Point)
        seen = set()
        learned_lits = []
        level_count = 0

        def analyze_literal(lit: Literal):
            nonlocal level_count
            var = lit.variable
            if self.level[var] == curr_level:
                level_count += 1
            elif self.level[var] > 0:
                learned_lits.append(lit)

        queue = [(lit, conflict)
                for lit in conflict.literals]

        while queue:
            lit, clause = queue.pop(0)
            var = lit.variable

            if var not in seen:
                seen.add(var)
                if self.level[var] == curr_level:
                    if var in self.antecedent:
                        parent = self.antecedent[var]
                        queue.extend(
                            (l, parent)
                            for l in parent.literals
                            if l.variable != var
                        )
                    else:
                        analyze_literal(lit)
                else:
                    learned_lits.append(lit)

        # Add learned clause
        if learned_lits:
            clause = Clause(learned_lits)
            self.learned_clauses.append(clause)

        # Return second highest level
        levels = sorted({self.level[lit.variable]
                        for lit in learned_lits})
        return levels[-2] if len(levels) > 1 else 0

    def _backtrack(self, level: int):
        """Backtrack to given level"""
        for var in list(self.assignment):
            if self.level[var] > level:
                del self.assignment[var]
                del self.level[var]
                if var in self.antecedent:
                    del self.antecedent[var]

    def _assign(self,
               var: int,
               value: bool,
               antecedent: Optional[Clause],
               level: int):
        """Make assignment with reason and level"""
        self.assignment[var] = value
        self.level[var] = level
        if antecedent is not None:
            self.antecedent[var] = antecedent

    def _clause_level(self, clause: Clause) -> int:
        """Get highest level in clause"""
        return max(self.level.get(lit.variable, 0)
                  for lit in clause.literals)

    def _choose_variable(self) -> int:
        """Choose next decision variable"""
        # Use VSIDS heuristic
        scores = defaultdict(float)
        decay = 0.95

        for clause in self.formula.clauses + self.learned_clauses:
            for lit in clause.literals:
                scores[lit.variable] *= decay
                scores[lit.variable] += 1

        unassigned = set(range(1, self.formula.num_vars + 1)) - \
                     set(self.assignment)
        return max(unassigned,
                  key=lambda v: scores[v])

class WalkSATSolver:
    """
    WalkSAT local search algorithm
    Incomplete but often effective
    """
    def __init__(self,
                formula: CNFFormula,
                max_flips: int = 1000,
                noise: float = 0.5):
        self.formula = formula
        self.max_flips = max_flips
        self.noise = noise

    def solve(self) -> Optional[Dict[int, bool]]:
        """Try to find satisfying assignment"""
        # Start with random assignment
        assignment = {
            var: random.choice([True, False])
            for var in range(1, self.formula.num_vars + 1)
        }

        best_assignment = assignment.copy()
        best_satisfied = self._count_satisfied(assignment)

        for _ in range(self.max_flips):
            if best_satisfied == len(self.formula.clauses):
                return best_assignment

            # Find unsatisfied clause
            unsat_clauses = [
                clause for clause in self.formula.clauses
                if not self._is_satisfied(clause, assignment)
            ]
            if not unsat_clauses:
                return assignment

            clause = random.choice(unsat_clauses)

            if random.random() < self.noise:
                # Random walk
                var = random.choice(
                    [lit.variable for lit in clause.literals]
                )
            else:
                # Greedy move
                var = min(
                    (lit.variable for lit in clause.literals),
                    key=lambda v: self._breaks_count(v, assignment)
                )

            # Flip variable
            assignment[var] = not assignment[var]

            # Update best solution
            satisfied = self._count_satisfied(assignment)
            if satisfied > best_satisfied:
                best_assignment = assignment.copy()
                best_satisfied = satisfied

        return best_assignment if best_satisfied == len(self.formula.clauses) else None

    def _is_satisfied(self,
                    clause: Clause,
                    assignment: Dict[int, bool]) -> bool:
        """Check if clause is satisfied"""
        return any(
            lit.positive == assignment[lit.variable]
            for lit in clause.literals
        )

    def _count_satisfied(self,
                       assignment: Dict[int, bool]) -> int:
        """Count satisfied clauses"""
        return sum(
            1 for clause in self.formula.clauses
            if self._is_satisfied(clause, assignment)
        )

    def _breaks_count(self,
                    var: int,
                    assignment: Dict[int, bool]) -> int:
        """Count clauses broken by flipping var"""
        count = 0
        assignment[var] = not assignment[var]  # Flip

        for clause in self.formula.clauses:
            was_sat = any(
                lit.positive == (not assignment[lit.variable])
                for lit in clause.literals
                if lit.variable == var
            )
            is_sat = self._is_satisfied(clause, assignment)
            if was_sat and not is_sat:
                count += 1

        assignment[var] = not assignment[var]  # Flip back
        return count

def compare_solvers(formula: CNFFormula):
    """Compare different SAT solving approaches"""
    solvers = [
        ("DPLL", DPLLSolver(formula)),
        ("CDCL", CDCLSolver(formula)),
        ("WalkSAT", WalkSATSolver(formula))
    ]

    results = {}
    print("\nComparing SAT Solvers:")
    print("-" * 40)

    for name, solver in solvers:
        start_time = time.time()
        assignment = solver.solve()
        elapsed = time.time() - start_time

        results[name] = {
            'assignment': assignment,
            'time': elapsed,
            'sat': assignment is not None,
            'verified': (assignment is not None and
                       formula.evaluate(assignment))
        }

        print(f"\n{name}:")
        print(f"SAT: {'Yes' if assignment else 'No'}")
        if assignment:
            print(f"Assignment: {assignment}")
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Verified: {results[name]['verified']}")

    return results

if __name__ == "__main__":
    # Example usage
    formula = CNFFormula([
        Clause([Literal(1, True), Literal(2, False)]),
        Clause([Literal(1, False), Literal(3, True)]),
        Clause([Literal(2, True), Literal(3, False)]),
        Clause([Literal(1, True), Literal(3, False)])
    ])

    print("Formula:", formula)
    results = compare_solvers(formula)

    # Additional analysis
    print("\nAnalysis:")
    print("1. DPLL:")
    print("   - Complete algorithm")
    print("   - Good for small instances")
    print("   - Uses unit propagation")

    print("\n2. CDCL:")
    print("   - Modern complete solver")
    print("   - Learns from conflicts")
    print("   - Better than DPLL in practice")

    print("\n3. WalkSAT:")
    print("   - Incomplete but fast")
    print("   - Good for random instances")
    print("   - May miss solutions")
