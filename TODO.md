# TODO

## Reuse primal shell linear-solve artifacts for VJPs

The direct RM shell forward solve currently assembles and factorizes the state
Jacobian for the primal solve, then discards those artifacts. At the end of the
same forward evaluation, `StateOperation.assemble_derivatives()` assembles the
BC-modified Jacobian again and creates a second MUMPS factorization for total
derivatives. This makes the forward evaluation pay for two tangent assemblies
and two factorizations, while subsequent VJPs only need inexpensive solves with
the stored derivative KSP.

Refactor the direct linear shell path to:

- retain the primal Jacobian matrix and configured/factorized PETSc KSP;
- pass those artifacts from the custom primal solve through `FEA.solve()` to
  `StateOperation`;
- reuse them for adjoint solves instead of assembling and factorizing the state
  Jacobian a second time;
- use `KSP.solveTranspose()` for the adjoint equation so the implementation does
  not rely on the shell tangent being symmetric;
- invalidate and replace the cached artifacts on every new primal evaluation;
- continue assembling the required input derivative matrices, `dR/dx`;
- verify that primal and derivative paths apply identical boundary-condition
  treatment, and add timing and derivative-consistency tests.

Start with the `custom_solve_direct` linear-shell path. Reuse from the generic
Newton path can be handled separately by retaining the converged Newton
Jacobian and solver.
