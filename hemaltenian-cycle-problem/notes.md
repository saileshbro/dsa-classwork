
# Hamiltonian Cycle Problem

## What is a Hamiltonian Cycle?

A **Hamiltonian cycle** in a graph is a cycle that:

- Visits **each vertex exactly once**.
- **Returns to the starting vertex**.

This means the cycle is **closed** and includes **every vertex** exactly once (except the starting vertex which is repeated at the end to close the loop).

---

## Formal Definition

Given a graph \( G = (V, E) \), where:

- \( V \) is the set of vertices,
- \( E \) is the set of edges,

A **Hamiltonian cycle** is a sequence of vertices:

```
v₁, v₂, v₃, ..., vₙ, v₁
```

Such that:

- Each vertex in \( V \) appears exactly once (except for \( v₁ \), which appears twice).
- Every consecutive pair \( (vᵢ, vᵢ₊₁) \in E \).
- The cycle returns to the starting vertex \( (vₙ, v₁) \in E \).

---

## Hamiltonian Path vs Hamiltonian Cycle

- **Hamiltonian Path**: Visits each vertex exactly once. No need to return to the start.
- **Hamiltonian Cycle**: A Hamiltonian path that returns to the starting point.

---

## Example 1: Graph With a Hamiltonian Cycle

Let:
- \( V = \{A, B, C, D, E\} \)
- \( E = \{(A,B), (B,C), (C,D), (D,E), (E,A), (B,D), (C,E)\} \)

Try the cycle:
`A → B → C → D → E → A`

- All vertices are visited exactly once.
- All edges exist.
- The cycle returns to A.

✅ This is a **Hamiltonian cycle**.

---

## Example 2: Graph Without a Hamiltonian Cycle

Let:
- \( V = \{A, B, C, D\} \)
- \( E = \{(A,B), (B,C), (C,D)\} \)

Try any cycle:

- No edge connects D to A.
- You can't return to the starting vertex.

❌ No Hamiltonian cycle exists.

---

## Why is This Problem Hard?

- The problem is **NP-complete**.
- It's in **NP**: You can verify a solution quickly.
- It's **NP-hard**: As hard as the hardest problems in NP.
- No known polynomial-time algorithm solves it for all cases.

---

## How Many Possibilities?

For a graph with \( n \) vertices:

- A **complete graph** has \( rac{(n-1)!}{2} \) possible Hamiltonian cycles.
- General graphs can have far fewer or none.
- Brute force search takes **exponential time**.

---

## Real-World Applications

- **Traveling Salesman Problem (TSP)**: Find the shortest Hamiltonian cycle.
- **Genome sequencing**: Reconstruct DNA from fragments.
- **Puzzle games**: E.g., Knight’s tour in chess.
- **Network design**: Build efficient routes that cover all nodes.

---

## Related Concepts

- **Eulerian Cycle**: Visits every **edge** exactly once.
- **Hamiltonian Cycle**: Visits every **vertex** exactly once.
- **TSP**: A Hamiltonian cycle with edge weights and cost minimization.

---

## Summary

- A Hamiltonian cycle visits all vertices once and returns.
- It’s **NP-complete** — hard to solve, easy to verify.
- Has applications in logistics, biology, and optimization.
- No efficient algorithm exists for all cases.

---

## Want to Try It in Code?

Let me know if you want a Python implementation to detect Hamiltonian cycles in small graphs.
