"""Matrix-free non-Hermitian TC solver using public PySCF FCI contractions."""

from __future__ import annotations

import numpy as np
from pyscf.fci import cistring, direct_nosym
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs


def _electron_tuple(nelec):
    if isinstance(nelec, (int, np.integer)):
        nelecb = int(nelec) // 2
        return int(nelec) - nelecb, nelecb
    return tuple(map(int, nelec))


def _dense_eigenvalues(operator, dimension):
    """Small-space fallback used only when ARPACK cannot be applied."""
    matrix = np.column_stack(
        [
            operator.matvec(np.eye(1, dimension, column, dtype=float).ravel())
            for column in range(dimension)
        ]
    )
    return np.linalg.eigvals(matrix)


def compute_fci_ground_state(h1, h2, norb, nelec, ecore=0.0, k_roots=1):
    """Return ``(E0, ci_dim, H_op)`` for a possibly non-Hermitian Hamiltonian.

    ``h2`` is in PySCF's chemists' ordering. The one-body Hamiltonian is first
    folded into it with ``fac=0.5``; ``H_op`` then applies only
    :func:`direct_nosym.contract_2e`, exactly as required by that interface.
    """
    h1 = np.asarray(h1, dtype=np.float64)
    eri = np.asarray(h2, dtype=np.float64)
    nelec = _electron_tuple(nelec)
    h2e_eff = direct_nosym.absorb_h1e(h1, eri, norb, nelec, fac=0.5)

    dim_a = cistring.num_strings(norb, nelec[0])
    dim_b = cistring.num_strings(norb, nelec[1])
    ci_dim = dim_a * dim_b

    def h_mv(c_flat):
        c = np.asarray(c_flat, dtype=np.float64).reshape(
            dim_a, dim_b, order="C"
        )
        hc = direct_nosym.contract_2e(h2e_eff, c, norb, nelec)
        if ecore != 0.0:
            hc = hc + ecore * c
        return np.asarray(hc).ravel(order="C")

    h_op = LinearOperator(
        (ci_dim, ci_dim), matvec=h_mv, dtype=np.float64
    )

    # ARPACK requires k < N - 1. Tiny CI spaces are better handled exactly.
    if ci_dim <= 2 or k_roots >= ci_dim - 1:
        evals = _dense_eigenvalues(h_op, ci_dim)
    else:
        try:
            evals, _ = eigs(
                h_op, k=k_roots, which="SR", maxiter=4000
            )
        except ArpackNoConvergence as error:
            if (
                error.eigenvalues is not None
                and len(error.eigenvalues) >= k_roots
            ):
                evals = error.eigenvalues
            else:
                try:
                    evals, _ = eigs(
                        h_op, k=k_roots, which="SM", maxiter=4000
                    )
                except ArpackNoConvergence:
                    # This tutorial uses a 25-dimensional helium CI space.
                    # A dense fallback makes failure deterministic without
                    # changing the primary matrix-free algorithm.
                    evals = _dense_eigenvalues(h_op, ci_dim)

    target = evals[np.argmin(evals.real)]
    return target, ci_dim, h_op


def tc_ground_energy(h, gt, nelec, e_nn):
    """Return the real smallest-real-part TC ground-state energy.

    ``gt`` uses physicists' ordering ``<pq|rs> = (pr|qs)_chem``.  PySCF's
    ``direct_nosym`` contractions take chemists' order and require the one-body
    term to be absorbed into the two-body tensor before ``contract_2e``.
    """
    h = np.ascontiguousarray(h, dtype=float)
    gt = np.ascontiguousarray(gt, dtype=float)
    norb = h.shape[0]
    eri_chem = np.ascontiguousarray(gt.transpose(0, 2, 1, 3))
    target, _, _ = compute_fci_ground_state(
        h, eri_chem, norb, nelec, ecore=e_nn
    )
    if abs(target.imag) > 1e-8:
        raise RuntimeError(
            f"selected TC root has a non-negligible imaginary part: {target}"
        )
    return float(target.real)


__all__ = ["compute_fci_ground_state", "tc_ground_energy"]
