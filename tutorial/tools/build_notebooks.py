"""Build the two src-independent tutorial notebooks with nbformat."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


TUTORIAL = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


PART1_CELLS = [
    md(
        r"""
        # Part 1: PySCF to MRA, plain integrals, and MRA-DMRG orbital optimization

        This notebook constructs the helium calculation without importing the
        repository research packages. It uses only the public PySCF, VAMPyR,
        NumPy/SciPy, and block2 APIs.

        The calculation has two stages:

        1. Project five PySCF RHF/cc-pVDZ molecular orbitals into an adaptive MRA
           representation and verify the plain one- and two-electron integrals.
        2. Optimize the real-space orbital shapes with the plain MRA-DMRG
           stationary equations, then save both the projected and optimized
           orbital families.

        All quantities are in Hartree atomic units. The nucleus is the exact
        point potential $V_{\mathrm{nuc}}(\mathbf r)=-2/|\mathbf r|$.
        """
    ),
    md(
        r"""
        ## Representation and precision

        A `FunctionTree` is **not a grid-value table**. Each leaf stores local
        polynomial coefficients. When a local wavelet detail is larger than the
        threshold implied by `PREC`, MRCPP replaces that three-dimensional cell
        by eight children. The polynomial order `ORDER` stays fixed; adaptivity
        changes the local refinement level.

        The tree can nevertheless be evaluated at any external Cartesian point.
        Its Cartesian derivatives are obtained with the MRA derivative operator,
        not by finite differences on a grid.
        """
    ),
    code(
        r"""
        import json
        import os
        import pathlib
        import sys
        import time

        here = pathlib.Path.cwd().resolve()
        if (here / "tutorial").is_dir():
            TUTORIAL = here / "tutorial"
        elif here.name == "tutorial":
            TUTORIAL = here
        else:
            raise RuntimeError("Run this notebook from the project root or tutorial/")
        os.chdir(TUTORIAL)
        sys.path.insert(0, str(TUTORIAL / "tools"))

        from env_provenance import write as write_provenance

        provenance = write_provenance(TUTORIAL / "data" / "provenance.json")
        print(json.dumps(provenance, indent=2, sort_keys=True))
        """
    ),
    code(
        r"""
        import numpy as np
        from scipy import linalg
        from pyscf import ao2mo, fci, gto, scf
        from vampyr import vampyr3d as vp

        np.set_printoptions(precision=8, suppress=True)

        BOX = 16.0
        ORDER = int(os.environ.get("MRA_TUTORIAL_ORDER", "11"))
        PREC = float(os.environ.get("MRA_TUTORIAL_PREC", "1e-5"))
        BASIS = "cc-pVDZ"
        N_ORBS = 5
        N_ELEC = 2
        NUCLEI = [(2.0, (0.0, 0.0, 0.0))]

        print(
            f"He/{BASIS}: {N_ORBS} orbitals, MRA box=[-{BOX:g},{BOX:g}]^3, "
            f"k={ORDER}, precision={PREC:g}"
        )
        """
    ),
    md(
        r"""
        ## The plain MRA operators

        The kinetic matrix is evaluated in gradient form,

        $$
        h_{pq}=\frac12\int\nabla\phi_p\cdot\nabla\phi_q\,d\mathbf r
        +\int \phi_pV_{\mathrm{nuc}}\phi_q\,d\mathbf r .
        $$

        Pair densities $\rho_{pr}=\phi_p\phi_r$ are convolved with the
        Poisson Green function. `PoissonOperator` applies $G_0$, so the source
        is explicitly $4\pi\rho$:

        $$
        g_{pqrs}=\langle pq|rs\rangle
        =\int \rho_{pr}(\mathbf r)\,
        G_0[4\pi\rho_{qs}](\mathbf r)\,d\mathbf r .
        $$

        Thus `g[p,q,r,s]` is in physicists' order and
        `g.transpose(0,2,1,3)` is in chemists' order.
        """
    ),
    code(
        r"""
        class MRAWorld:
            def __init__(self, nuclei, box, order, prec):
                self.nuclei = list(nuclei)
                self.box = float(box)
                self.order = int(order)
                self.prec = float(prec)
                self.mra = vp.MultiResolutionAnalysis(
                    box=[-int(box), int(box)], order=order
                )
                self.projector = vp.ScalingProjector(self.mra, prec=prec)
                self.poisson = vp.PoissonOperator(self.mra, prec=prec)
                self.derivative = vp.ABGVDerivative(self.mra, a=0.5, b=0.5)
                self._helmholtz = {}

                def point_nuclear_potential(r):
                    value = 0.0
                    for charge, center in self.nuclei:
                        distance = np.linalg.norm(np.asarray(r) - np.asarray(center))
                        value -= charge / max(float(distance), 1e-12)
                    return value

                self.v_nuc = self.projector(point_nuclear_potential)
                self.e_nn = sum(
                    za * zb / np.linalg.norm(np.asarray(ra) - np.asarray(rb))
                    for a, (za, ra) in enumerate(self.nuclei)
                    for b, (zb, rb) in enumerate(self.nuclei)
                    if a < b
                )

            def project(self, function):
                return self.projector(function)

            @staticmethod
            def dot(left, right):
                return vp.dot(left, right)

            def linear_combination(self, coefficients, trees):
                terms = [
                    (float(coefficient), tree)
                    for coefficient, tree in zip(coefficients, trees)
                    if abs(coefficient) > 1e-14
                ]
                if not terms:
                    return 0.0 * trees[0]
                result = vp.FunctionTree(self.mra)
                vp.advanced.add(self.prec, result, terms)
                result.crop(self.prec)
                return result

            def lowdin(self, orbitals):
                overlap = np.array(
                    [[self.dot(a, b) for b in orbitals] for a in orbitals]
                )
                eigenvalues, eigenvectors = np.linalg.eigh(overlap)
                if eigenvalues.min() < 1e-10:
                    raise RuntimeError(
                        f"Nearly linearly dependent orbitals: min eig(S)="
                        f"{eigenvalues.min():.3e}"
                    )
                inverse_half = (
                    eigenvectors
                    @ np.diag(eigenvalues ** -0.5)
                    @ eigenvectors.T
                )
                return [
                    self.linear_combination(inverse_half[:, p], orbitals)
                    for p in range(len(orbitals))
                ]

            def pair_densities(self, orbitals):
                rho = {}
                for p in range(len(orbitals)):
                    for q in range(p, len(orbitals)):
                        value = orbitals[p] * orbitals[q]
                        value.crop(self.prec)
                        rho[(p, q)] = value
                return rho

            def pair_potentials(self, rho):
                potentials = {}
                for pair, density in rho.items():
                    value = self.poisson(4.0 * np.pi * density)
                    value.crop(self.prec)
                    potentials[pair] = value
                return potentials

            def one_body(self, orbitals, rho):
                gradients = [
                    vp.gradient(self.derivative, orbital) for orbital in orbitals
                ]
                matrix = np.empty((len(orbitals), len(orbitals)))
                for p in range(len(orbitals)):
                    for q in range(p, len(orbitals)):
                        kinetic = 0.5 * sum(
                            self.dot(gradients[p][axis], gradients[q][axis])
                            for axis in range(3)
                        )
                        nuclear = self.dot(rho[(p, q)], self.v_nuc)
                        matrix[p, q] = matrix[q, p] = kinetic + nuclear
                return matrix, gradients

            def two_body(self, rho, potentials, norb):
                pairs = sorted(rho)
                pair_values = {}
                for p_index, p_pair in enumerate(pairs):
                    for q_pair in pairs[p_index:]:
                        value = self.dot(rho[p_pair], potentials[q_pair])
                        pair_values[(p_pair, q_pair)] = value
                        pair_values[(q_pair, p_pair)] = value

                tensor = np.empty((norb, norb, norb, norb))
                for p in range(norb):
                    for q in range(norb):
                        for r in range(norb):
                            for s in range(norb):
                                pr = tuple(sorted((p, r)))
                                qs = tuple(sorted((q, s)))
                                tensor[p, q, r, s] = pair_values[(pr, qs)]
                return tensor

            def integrals(self, orbitals):
                rho = self.pair_densities(orbitals)
                potentials = self.pair_potentials(rho)
                h, gradients = self.one_body(orbitals, rho)
                g = self.two_body(rho, potentials, len(orbitals))
                return h, g, rho, potentials, gradients

            def helmholtz(self, exponent):
                key = round(float(exponent), 12)
                if key not in self._helmholtz:
                    self._helmholtz[key] = vp.HelmholtzOperator(
                        self.mra, exp=exponent, prec=self.prec
                    )
                return self._helmholtz[key]

        ctx = MRAWorld(NUCLEI, BOX, ORDER, PREC)
        print("MRA world and point nuclear potential constructed.")
        """
    ),
    md(
        r"""
        ## PySCF reference and projection

        PySCF supplies the initial RHF orbitals and an independent analytic
        Gaussian-integral reference. The initialization follows the same
        hierarchy as the Gaussian basis:

        $$
        \chi_A(\mathbf r)=\sum_a d_{Aa}g_a(\mathbf r),
        \qquad
        \phi_p(\mathbf r)=\sum_A C_{Ap}\chi_A(\mathbf r).
        $$

        `ScalingProjector` is linear, so the code projects each unique analytic
        primitive once and then performs the two linear combinations in tree
        form:

        $$
        P_{\mathrm{MRA}}\phi_p
        =
        \sum_A C_{Ap}
        \underbrace{\sum_a d_{Aa}P_{\mathrm{MRA}}g_a}_{\text{AO tree }A}.
        $$

        The He/cc-pVDZ basis contains only $s$ and $p$ shells. A VAMPyR
        `GaussFunc` represents an unnormalized Cartesian monomial, so the
        PySCF real-spherical angular factors are inserted exactly:

        $$
        N_s=\frac{1}{2\sqrt{\pi}},
        \qquad
        N_p=\sqrt{\frac{3}{4\pi}}.
        $$

        No fixed grid of MO values is built. `ScalingProjector` evaluates the
        analytic primitives only on its adaptive cell quadrature. The
        `primitive_trees` and `ao_trees` below are temporary initialization
        objects. After Löwdin orthonormalization, both the pure checkpoint and
        the orbital optimizer contain only MO trees. A basis containing $d$ or
        higher shells requires the exact real-spherical-to-Cartesian
        transformation and is rejected by this tutorial implementation.
        """
    ),
    code(
        r"""
        mol = gto.M(
            atom=[("He", (0.0, 0.0, 0.0))],
            basis=BASIS,
            unit="Bohr",
            charge=0,
            spin=0,
            verbose=0,
        )
        mean_field = scf.RHF(mol)
        e_rhf = float(mean_field.kernel())
        if not mean_field.converged:
            raise RuntimeError("PySCF RHF did not converge")

        coefficients = np.asarray(mean_field.mo_coeff[:, :N_ORBS])
        h_ao = mean_field.get_hcore()
        h_reference = coefficients.T @ h_ao @ coefficients
        eri_reference = ao2mo.restore(
            1, ao2mo.kernel(mol, coefficients), N_ORBS
        )
        g_reference = eri_reference.transpose(0, 2, 1, 3)
        e_fci_reference, _ = fci.direct_spin1.kernel(
            h_reference, eri_reference, N_ORBS, N_ELEC
        )

        print(f"PySCF RHF energy       = {e_rhf:.10f} Ha")
        print(f"PySCF FCI/cc-pVDZ      = {e_fci_reference:.10f} Ha")
        print(f"Exact nonrelativistic  = {-2.903724:.10f} Ha")
        """
    ),
    code(
        r"""
        _POLYNOMIAL = {
            "": (0, 0, 0),
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),
        }

        def ao_primitive_expansions(molecule):
            from pyscf.gto import gto_norm

            angular_factor = {
                0: 1.0 / (2.0 * np.sqrt(np.pi)),
                1: np.sqrt(3.0 / (4.0 * np.pi)),
            }
            labels = molecule.ao_labels(fmt=None)
            expansions = []
            ao_index = 0
            for shell in range(molecule.nbas):
                angular_momentum = molecule.bas_angular(shell)
                if angular_momentum > 1:
                    raise NotImplementedError(
                        "The fast tutorial projector supports only s and p shells"
                    )
                exponents = molecule.bas_exp(shell)
                contractions = (
                    molecule.bas_ctr_coeff(shell)
                    * gto_norm(angular_momentum, exponents)[:, None]
                    * angular_factor[angular_momentum]
                )
                center = tuple(float(x) for x in molecule.bas_coord(shell))
                for contraction in range(contractions.shape[1]):
                    for _component in range(2 * angular_momentum + 1):
                        polynomial = _POLYNOMIAL[labels[ao_index][3]]
                        expansions.append(
                            [
                                (
                                    float(exponent),
                                    float(coefficient),
                                    center,
                                    polynomial,
                                )
                                for exponent, coefficient in zip(
                                    exponents, contractions[:, contraction]
                                )
                            ]
                        )
                        ao_index += 1
            assert ao_index == molecule.nao
            return expansions

        def project_molecular_orbitals(context, molecule, mo_coefficients):
            ao_expansions = ao_primitive_expansions(molecule)

            # Stage 1: project every unique analytic primitive once.
            primitive_trees = {}
            for expansion in ao_expansions:
                for exponent, _, center, polynomial in expansion:
                    key = (exponent, center, polynomial)
                    if key not in primitive_trees:
                        gaussian = vp.GaussFunc(
                            beta=exponent,
                            alpha=1.0,
                            position=list(center),
                            poly_exponent=list(polynomial),
                        )
                        primitive_trees[key] = context.project(gaussian)

            # Stage 2: form each contracted spherical AO as an MRA tree.
            ao_trees = []
            for expansion in ao_expansions:
                terms = [
                    (
                        coefficient,
                        primitive_trees[(exponent, center, polynomial)],
                    )
                    for exponent, coefficient, center, polynomial in expansion
                ]
                ao_tree = vp.FunctionTree(context.mra)
                vp.advanced.add(context.prec, ao_tree, terms)
                ao_tree.crop(context.prec)
                ao_trees.append(ao_tree)

            # Stage 3: form PySCF molecular orbitals by the LCAO coefficients.
            mo_trees = []
            for orbital in range(mo_coefficients.shape[1]):
                terms = [
                    (
                        float(mo_coefficients[ao, orbital]),
                        ao_trees[ao],
                    )
                    for ao in range(len(ao_trees))
                    if abs(mo_coefficients[ao, orbital]) > 1e-14
                ]
                tree = vp.FunctionTree(context.mra)
                vp.advanced.add(context.prec, tree, terms)
                tree.crop(context.prec)
                mo_trees.append(tree)

            # Löwdin corrects only finite-precision projection error. From this
            # point onward every active and persisted FunctionTree is an MO.
            projected_mos = context.lowdin(mo_trees)
            return projected_mos

        started = time.time()
        pure_orbitals = project_molecular_orbitals(ctx, mol, coefficients)
        overlap_pure = np.array(
            [[ctx.dot(a, b) for b in pure_orbitals] for a in pure_orbitals]
        )
        print(f"Projected and orthonormalized in {time.time() - started:.1f} s")
        print(
            "max|S-I| =",
            np.max(np.abs(overlap_pure - np.eye(N_ORBS))),
        )
        """
    ),
    code(
        r"""
        started = time.time()
        h_mra, g_mra, rho_pure, potentials_pure, gradients_pure = ctx.integrals(
            pure_orbitals
        )
        delta_h = float(np.max(np.abs(h_mra - h_reference)))
        delta_g = float(np.max(np.abs(g_mra - g_reference)))
        e_fci_mra, _ = fci.direct_spin1.kernel(
            h_mra,
            g_mra.transpose(0, 2, 1, 3),
            N_ORBS,
            N_ELEC,
        )
        delta_energy = float(abs(e_fci_mra - e_fci_reference))

        print(f"Plain MRA integrals built in {time.time() - started:.1f} s")
        print(f"max|h_MRA-h_PySCF| = {delta_h:.3e}")
        print(f"max|g_MRA-g_PySCF| = {delta_g:.3e}")
        print(f"E_FCI[MRA h,g]      = {e_fci_mra:.10f} Ha")
        print(f"|Delta E_FCI|       = {delta_energy:.3e} Ha")

        # The elementwise target comes from the tutorial specification.
        # At k=9, PREC=1e-5 gave max|Delta g|=2.19e-7. The executable default
        # therefore raises the local polynomial order to k=11 while retaining
        # the documented 1e-5 adaptive threshold.
        assert delta_h <= 1e-7, delta_h
        assert delta_g <= 1e-7, delta_g
        """
    ),
    md(
        r"""
        ## Saving a reproducible orbital checkpoint

        `saveTree` stores the adaptive coefficients, not samples on a shared
        grid. `world.json` records the MRA domain, order, precision, nuclear
        model, orbital order, and filenames needed by Part 2.
        """
    ),
    code(
        r"""
        def overlap_matrix(context, orbitals):
            return np.array(
                [[context.dot(a, b) for b in orbitals] for a in orbitals]
            )

        def save_orbital_family(context, orbitals, family, occupations, metadata):
            directory = TUTORIAL / "data" / f"he_ccpvdz_{family}"
            directory.mkdir(parents=True, exist_ok=True)
            records = []
            for index, orbital in enumerate(orbitals):
                basename = f"mo_{index:03d}"
                returned = pathlib.Path(
                    orbital.saveTree(filename=str(directory / basename))
                )
                records.append(
                    {
                        "index": index,
                        "basename": basename,
                        "tree": returned.name,
                        "occ": float(occupations[index]),
                    }
                )
            world = {
                "box": context.box,
                "order": context.order,
                "prec": context.prec,
                "nuclei": [
                    [float(z), [float(x) for x in center]]
                    for z, center in context.nuclei
                ],
                "nuc_model": "point",
                "basis": BASIS,
                "nelec": N_ELEC,
                "orbital_family": family,
                "orbitals": records,
                "overlap": overlap_matrix(context, orbitals).tolist(),
                "provenance_file": "../provenance.json",
                **metadata,
            }
            with (directory / "world.json").open("w") as handle:
                json.dump(world, handle, indent=2, sort_keys=True)
            return directory / "world.json"

        pure_world = save_orbital_family(
            ctx,
            pure_orbitals,
            "pure",
            [2, 0, 0, 0, 0],
            {
                "e_rhf": e_rhf,
                "e_fci_plain": float(e_fci_mra),
                "max_delta_h": delta_h,
                "max_delta_g": delta_g,
            },
        )
        print("Saved", pure_world.relative_to(TUTORIAL))
        """
    ),
    md(
        r"""
        ## MRA-DMRG optimization

        At fixed orbitals, block2 returns the electronic energy and spin-summed
        one- and two-particle density matrices. With

        $$
        H=\sum h_{pq}a_p^\dagger a_q+
        \frac12\sum g_{pqrs}a_p^\dagger a_q^\dagger a_sa_r,
        $$

        the energy derivatives are
        $\partial E/\partial h=\gamma$ and
        $\partial E/\partial g=\Gamma/2$. The Euler identity

        $$
        E_{\mathrm{elec}}=\langle h,\partial E/\partial h\rangle+
        \langle g,\partial E/\partial g\rangle
        $$

        checks the factor and index convention on every macro-iteration.
        The resulting stationary orbital equation is inverted with a bound-state
        `HelmholtzOperator`, followed by Loewdin orthonormalization.
        """
    ),
    code(
        r"""
        class Block2Solver:
            def __init__(
                self,
                nelec,
                bond_dim=250,
                num_sweeps=20,
                tolerance=1e-12,
            ):
                self.nelec = int(nelec)
                self.bond_dim = int(bond_dim)
                self.num_sweeps = int(num_sweeps)
                self.tolerance = float(tolerance)
                self.calls = 0
                self.driver = None

            def __call__(self, h, g):
                from pyblock2.driver.core import DMRGDriver, SymmetryTypes

                norb = h.shape[0]
                if self.driver is None:
                    self.driver = DMRGDriver(
                        scratch=str(TUTORIAL / "tmp_block2"),
                        symm_type=SymmetryTypes.SU2,
                        n_threads=1,
                    )
                self.driver.initialize_system(
                    n_sites=norb, n_elec=self.nelec, spin=0
                )
                eri_chem = np.ascontiguousarray(
                    g.transpose(0, 2, 1, 3), dtype=float
                )
                mpo = self.driver.get_qc_mpo(
                    h1e=np.ascontiguousarray(h, dtype=float),
                    g2e=eri_chem,
                    ecore=0.0,
                    iprint=0,
                )
                self.calls += 1
                ket = self.driver.get_random_mps(
                    tag=f"KET{self.calls}",
                    bond_dim=self.bond_dim,
                    nroots=1,
                )
                sweeps = self.num_sweeps
                bond_dims = [min(50, self.bond_dim)] * min(4, sweeps)
                bond_dims += [self.bond_dim] * (sweeps - len(bond_dims))
                noises = [1e-4] * min(4, sweeps)
                noises += [1e-6] * min(4, sweeps - len(noises))
                noises += [0.0] * (sweeps - len(noises))
                energy = float(
                    self.driver.dmrg(
                        mpo,
                        ket,
                        n_sweeps=sweeps,
                        bond_dims=bond_dims,
                        noises=noises,
                        thrds=[self.tolerance] * sweeps,
                        tol=self.tolerance,
                        iprint=0,
                    )
                )
                dh = np.asarray(self.driver.get_1pdm(ket), dtype=float)
                dg = 0.5 * np.asarray(
                    self.driver.get_2pdm(ket), dtype=float
                )
                euler = float(np.sum(h * dh) + np.sum(g * dg))
                assert abs(energy - euler) <= 1e-7 * max(1.0, abs(energy)), (
                    energy,
                    euler,
                )
                return energy, dh, dg

        def symmetrize_energy_derivatives(dh, dg):
            d1 = 0.5 * (dh + dh.T)
            d2 = 0.25 * (
                dg
                + dg.transpose(1, 0, 3, 2)
                + dg.transpose(2, 1, 0, 3)
                + dg.transpose(1, 2, 3, 0)
            )
            return d1, d2

        def lagrange_multipliers(d1, d2, h, g):
            epsilon = np.einsum("mj,nj->mn", d1, h)
            epsilon += 2.0 * np.einsum("mjkl,njkl->mn", d2, g)
            return 0.5 * (epsilon + epsilon.T)
        """
    ),
    code(
        r"""
        def update_orbitals(context, orbitals, potentials, rotation, occupations,
                            epsilon_rotated, d2):
            norb = len(orbitals)
            rotated = [
                context.linear_combination(rotation[p, :], orbitals)
                for p in range(norb)
            ]
            updated = []
            for p in range(norb):
                if abs(occupations[p]) < 1e-10:
                    print(f"  [warn] occupation {p} is too small; orbital frozen")
                    updated.append(rotated[p])
                    continue
                ratio = epsilon_rotated[p, p] / occupations[p]
                if ratio >= 0.0:
                    print(
                        f"  [warn] epsilon/lambda={ratio:.4e} for orbital {p}; "
                        "orbital frozen"
                    )
                    updated.append(rotated[p])
                    continue

                exponent = np.sqrt(-2.0 * ratio)
                amplitude = np.einsum("i,ijkl->jkl", rotation[p, :], d2)
                rhs = context.v_nuc * rotated[p]
                rhs.crop(context.prec)

                for k in range(norb):
                    coefficients_by_pair = {}
                    for j in range(norb):
                        for l in range(norb):
                            pair = tuple(sorted((j, l)))
                            coefficients_by_pair[pair] = (
                                coefficients_by_pair.get(pair, 0.0)
                                + amplitude[j, k, l]
                            )
                    weighted_potential = None
                    for pair, coefficient in coefficients_by_pair.items():
                        if abs(coefficient) < 1e-12:
                            continue
                        term = coefficient * potentials[pair]
                        weighted_potential = (
                            term
                            if weighted_potential is None
                            else weighted_potential + term
                        )
                    if weighted_potential is not None:
                        term = weighted_potential * orbitals[k]
                        term.crop(context.prec)
                        rhs = rhs + (2.0 / occupations[p]) * term

                for q in range(norb):
                    if q == p or abs(epsilon_rotated[p, q]) < 1e-12:
                        continue
                    rhs = rhs - (
                        epsilon_rotated[p, q] / occupations[p]
                    ) * rotated[q]

                rhs.crop(context.prec)
                new_orbital = -2.0 * context.helmholtz(exponent)(rhs)
                new_orbital.crop(context.prec)
                updated.append(new_orbital)
            return context.lowdin(updated)

        def optimize_orbitals(context, initial_orbitals, solver, delta=1e-5,
                              max_iter=40):
            orbitals = context.lowdin(initial_orbitals)
            history = []
            previous = None
            for iteration in range(1, max_iter + 1):
                started = time.time()
                h, g, _, potentials, _ = context.integrals(orbitals)
                electronic, dh, dg = solver(h, g)
                total = electronic + context.e_nn
                history.append(total)
                change = np.nan if previous is None else total - previous
                print(
                    f"iter {iteration:2d}: E={total:.10f}  "
                    f"dE={change:+.3e}  time={time.time()-started:.1f}s"
                )
                if previous is not None and abs(change) < delta:
                    return total, orbitals, history
                previous = total

                d1, d2 = symmetrize_energy_derivatives(dh, dg)
                epsilon = lagrange_multipliers(d1, d2, h, g)
                occupations, vectors = np.linalg.eigh(d1)
                order = np.argsort(occupations)[::-1]
                occupations = occupations[order]
                vectors = vectors[:, order]
                rotation = vectors.T
                epsilon_rotated = rotation @ epsilon @ rotation.T
                print("  occupations:", np.array2string(occupations, precision=5))
                orbitals = update_orbitals(
                    context,
                    orbitals,
                    potentials,
                    rotation,
                    occupations,
                    epsilon_rotated,
                    d2,
                )
            print("WARNING: maximum number of macro-iterations reached")
            return previous, orbitals, history
        """
    ),
    code(
        r"""
        solver = Block2Solver(
            N_ELEC,
            bond_dim=int(os.environ.get("MRA_TUTORIAL_BOND_DIM", "250")),
            num_sweeps=int(os.environ.get("MRA_TUTORIAL_SWEEPS", "20")),
        )
        e_optimized, optimized_orbitals, optimization_history = optimize_orbitals(
            ctx,
            pure_orbitals,
            solver,
            delta=float(os.environ.get("MRA_TUTORIAL_DELTA", "1e-5")),
            max_iter=int(os.environ.get("MRA_TUTORIAL_MAX_ITER", "40")),
        )
        print(f"Optimized MRA-DMRG energy = {e_optimized:.10f} Ha")
        print("Reference reported for He/cc-pVDZ: -2.89767133 Ha")
        assert abs(e_optimized - (-2.89767133)) < 5e-4, e_optimized

        h_optimized, g_optimized, _, _, _ = ctx.integrals(optimized_orbitals)
        optimized_world = save_orbital_family(
            ctx,
            optimized_orbitals,
            "optimized",
            np.linalg.eigvalsh(
                solver(h_optimized, g_optimized)[1]
            )[::-1],
            {
                "e_mra_dmrg": float(e_optimized),
                "convergence_history": [float(x) for x in optimization_history],
            },
        )
        print("Saved", optimized_world.relative_to(TUTORIAL))
        """
    ),
    md(
        r"""
        ## Part 1 checkpoint

        Part 2 needs only `world.json` and the `.tree` files. The initial
        Gaussian coefficients are not needed after projection. The optimized
        trees are also not assumed to remain in the original Gaussian span.
        """
    ),
]


PART2_CELLS = [
    md(
        r"""
        # Part 2: Fixed-orbital erfc transcorrelation

        This notebook reloads the MRA trees saved by Part 1 and assembles the
        helium transcorrelated Hamiltonian. The explanatory demo runs
        $\mu\in\{0.5,1.0\}$ by default for both projected and plain
        MRA-DMRG-optimized orbital families. The additional full-scan values
        remain visible and commented out in the parameter cell.

        The orbitals remain adaptive MRA functions. Only the radial Jastrow
        kernels are converted to finite Gaussian sums. Helium has two electrons,
        so its explicit three-body term is exactly zero.
        """
    ),
    md(
        r"""
        ## Exact identities and numerical approximations

        The analytic erfc Jastrow identities are exact:

        $$
        u'(r)=\tfrac12\operatorname{erfc}(\mu r),\qquad
        q_1(r)=\frac{\operatorname{erfc}(\mu r)}{2r},
        $$

        $$
        q_2(r)=\nabla^2u=
        \frac{\operatorname{erfc}(\mu r)}{r}
        -\frac{\mu}{\sqrt{\pi}}e^{-\mu^2r^2},\qquad
        q_3(r)=\frac{\operatorname{erfc}(\mu r)^2}{4r^2}.
        $$

        The finite Gaussian representation of the erfc parts is the numerical
        approximation. This notebook uses GL24 as the production
        cost/accuracy compromise. GL32 or GL64 can be selected by increasing
        `GL_ORDER`; GL64 is the tighter kernel check. In $q_2$, the negative
        $e^{-\mu^2r^2}$ term is exact. Only its erfc-over-$r$ part is
        discretized.

        MRCPP convolution operators consume separated Gaussian kernels. This
        requirement, rather than the form of the molecular orbitals, is why the
        radial kernels are approximated. The MO `FunctionTree` is **not a
        grid-value table** and is never fitted to Gaussian orbitals.
        """
    ),
    code(
        r"""
        import json
        import os
        import pathlib
        import sys
        import time

        here = pathlib.Path.cwd().resolve()
        if (here / "tutorial").is_dir():
            TUTORIAL = here / "tutorial"
        elif here.name == "tutorial":
            TUTORIAL = here
        else:
            raise RuntimeError("Run this notebook from the project root or tutorial/")
        os.chdir(TUTORIAL)
        sys.path.insert(0, str(TUTORIAL))
        sys.path.insert(0, str(TUTORIAL / "tools"))

        import numpy as np
        from pyscf import fci
        from vampyr import vampyr1d as vp1
        from vampyr import vampyr3d as vp

        import helper
        from env_provenance import write as write_provenance
        from tc_solve import tc_ground_energy

        provenance = write_provenance(TUTORIAL / "data" / "provenance.json")
        GL_ORDER = int(os.environ.get("MRA_TUTORIAL_GL_ORDER", "24"))
        X_MIN = 8.7e-5
        MU_VALUES = (
            0.5,
            # 0.7,
            1.0,
            # 1.5,
            # 2.0,
        )
        print(f"GL order = {GL_ORDER}; mu scan = {MU_VALUES}")
        print("GL24:", helper.kernel_errors(1.0, X_MIN, 24))
        print("GL64:", helper.kernel_errors(1.0, X_MIN, 64))
        """
    ),
    md(
        r"""
        ## Reloading adaptive orbital trees

        The MRA world must be reconstructed before `loadTree` is called. A tree
        is meaningful only together with its box, polynomial order, precision,
        and MRCPP/VAMPyR build. Different orbitals may have different adaptive
        leaves; there is no requirement that they share a nodal grid.
        """
    ),
    code(
        r"""
        class FixedMRAWorld:
            def __init__(self, metadata):
                self.box = float(metadata["box"])
                self.order = int(metadata["order"])
                self.prec = float(metadata["prec"])
                self.nuclei = [
                    (float(z), tuple(float(x) for x in center))
                    for z, center in metadata["nuclei"]
                ]
                self.mra = vp.MultiResolutionAnalysis(
                    box=[-int(self.box), int(self.box)],
                    order=self.order,
                )
                self.poisson = vp.PoissonOperator(self.mra, prec=self.prec)
                self.derivative = vp.ABGVDerivative(
                    self.mra, a=0.5, b=0.5
                )
                projector = vp.ScalingProjector(self.mra, prec=self.prec)

                def point_nuclear_potential(r):
                    value = 0.0
                    for charge, center in self.nuclei:
                        distance = np.linalg.norm(
                            np.asarray(r) - np.asarray(center)
                        )
                        value -= charge / max(float(distance), 1e-12)
                    return value

                self.v_nuc = projector(point_nuclear_potential)
                self.e_nn = sum(
                    za * zb / np.linalg.norm(np.asarray(ra) - np.asarray(rb))
                    for a, (za, ra) in enumerate(self.nuclei)
                    for b, (zb, rb) in enumerate(self.nuclei)
                    if a < b
                )

            @staticmethod
            def dot(left, right):
                return vp.dot(left, right)

            def pair_densities(self, orbitals):
                rho = {}
                for p in range(len(orbitals)):
                    for q in range(p, len(orbitals)):
                        value = orbitals[p] * orbitals[q]
                        value.crop(self.prec)
                        rho[(p, q)] = value
                return rho

            def plain_intermediates(self, orbitals):
                rho = self.pair_densities(orbitals)
                potentials = {}
                for pair, density in rho.items():
                    value = self.poisson(4.0 * np.pi * density)
                    value.crop(self.prec)
                    potentials[pair] = value
                gradients = [
                    vp.gradient(self.derivative, orbital)
                    for orbital in orbitals
                ]
                h = np.empty((len(orbitals), len(orbitals)))
                for p in range(len(orbitals)):
                    for q in range(p, len(orbitals)):
                        kinetic = 0.5 * sum(
                            self.dot(gradients[p][axis], gradients[q][axis])
                            for axis in range(3)
                        )
                        nuclear = self.dot(rho[(p, q)], self.v_nuc)
                        h[p, q] = h[q, p] = kinetic + nuclear

                pairs = sorted(rho)
                table = {}
                for p_index, p_pair in enumerate(pairs):
                    for q_pair in pairs[p_index:]:
                        value = self.dot(rho[p_pair], potentials[q_pair])
                        table[(p_pair, q_pair)] = value
                        table[(q_pair, p_pair)] = value
                norb = len(orbitals)
                g = np.empty((norb, norb, norb, norb))
                for p in range(norb):
                    for q in range(norb):
                        for r in range(norb):
                            for s in range(norb):
                                g[p, q, r, s] = table[
                                    (
                                        tuple(sorted((p, r))),
                                        tuple(sorted((q, s))),
                                    )
                                ]
                return {
                    "rho": rho,
                    "pairs": pairs,
                    "gradients": gradients,
                    "h": h,
                    "g": g,
                }

        def load_orbital_family(family):
            directory = TUTORIAL / "data" / f"he_ccpvdz_{family}"
            with (directory / "world.json").open() as handle:
                metadata = json.load(handle)
            context = FixedMRAWorld(metadata)
            orbitals = []
            for record in metadata["orbitals"]:
                tree = vp.FunctionTree(context.mra)
                tree.loadTree(filename=str(directory / record["basename"]))
                orbitals.append(tree)
            overlap = np.array(
                [[context.dot(a, b) for b in orbitals] for a in orbitals]
            )
            assert np.max(np.abs(overlap - np.eye(len(orbitals)))) < 1e-6
            return context, orbitals, metadata

        families = {}
        for family in ("pure", "optimized"):
            context, orbitals, metadata = load_orbital_family(family)
            started = time.time()
            plain = context.plain_intermediates(orbitals)
            families[family] = {
                "context": context,
                "orbitals": orbitals,
                "metadata": metadata,
                "plain": plain,
            }
            print(
                f"{family}: loaded {len(orbitals)} trees and built "
                f"mu-independent intermediates in {time.time()-started:.1f} s"
            )
        """
    ),
    md(
        r"""
        ## VAMPyR-style convolution operators

        `K1` uses the three power-one Cartesian kernels and MRA orbital
        derivatives. `K2` is a direct scalar convolution. The current backend
        drops negative Gaussian coefficients, so its positive and negative
        pieces are applied separately and subtracted. `K3` sums the three
        power-two Cartesian convolutions to reconstruct
        $\Delta x^2+\Delta y^2+\Delta z^2=r_{12}^2$.
        """
    ),
    code(
        r"""
        def gauss_exp(terms):
            expansion = vp1.GaussExp()
            for coefficient, exponent in terms:
                if coefficient <= 0.0:
                    raise ValueError("gauss_exp accepts positive terms only")
                expansion.append(
                    vp1.GaussFunc(float(exponent), float(coefficient))
                )
            return expansion

        def scalar_convolution(context, terms):
            positive = [(c, z) for c, z in terms if c > 0.0]
            negative = [(-c, z) for c, z in terms if c < 0.0]
            positive_operator = (
                vp.ConvolutionOperator(
                    context.mra, gauss_exp(positive), context.prec
                )
                if positive
                else None
            )
            negative_operator = (
                vp.ConvolutionOperator(
                    context.mra, gauss_exp(negative), context.prec
                )
                if negative
                else None
            )

            def apply(density):
                result = (
                    positive_operator(density)
                    if positive_operator is not None
                    else 0.0 * density
                )
                if negative_operator is not None:
                    result = result - negative_operator(density)
                result.crop(context.prec)
                return result

            return apply

        def cartesian_convolution(context, terms):
            return vp.CartesianConvolution(
                context.mra, gauss_exp(terms), context.prec
            )

        def vector_field(operator, density, precision):
            components = []
            for powers in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                operator.setCartesianComponents(*powers)
                value = operator(density)
                value.crop(precision)
                components.append(value)
            return components

        def r2_scalar(operator, density, precision):
            result = None
            for powers in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
                operator.setCartesianComponents(*powers)
                value = operator(density)
                result = value if result is None else result + value
            result.crop(precision)
            return result

        def pair_index_data(norb):
            pairs = [(p, q) for p in range(norb) for q in range(p, norb)]
            lookup = {pair: index for index, pair in enumerate(pairs)}
            pair_index = np.empty((norb, norb), dtype=np.intp)
            for p in range(norb):
                for q in range(norb):
                    pair_index[p, q] = lookup[tuple(sorted((p, q)))]
            return pairs, pair_index
        """
    ),
    code(
        r"""
        def scalar_pair_tensor(rho, fields, pairs, pair_index):
            table = np.empty((len(pairs), len(pairs)))
            for p_index, p_pair in enumerate(pairs):
                for q_index in range(p_index, len(pairs)):
                    q_pair = pairs[q_index]
                    value = vp.dot(rho[p_pair], fields[q_pair])
                    table[p_index, q_index] = table[q_index, p_index] = value
            return np.ascontiguousarray(
                table[
                    pair_index[:, None, :, None],
                    pair_index[None, :, None, :],
                ]
            )

        def drift_tensor(orbitals, fields, pairs, pair_index, gradients, precision):
            norb = len(orbitals)
            contractions = np.empty((len(pairs), norb, norb))
            for pair_number, pair in enumerate(pairs):
                for differentiated in range(norb):
                    product = None
                    for axis in range(3):
                        term = fields[pair][axis] * gradients[differentiated][axis]
                        product = term if product is None else product + term
                    product.crop(precision)
                    for bra in range(norb):
                        contractions[pair_number, differentiated, bra] = vp.dot(
                            orbitals[bra], product
                        )
            index = np.arange(norb, dtype=np.intp)
            p = index[:, None, None, None]
            q = index[None, :, None, None]
            r = index[None, None, :, None]
            s = index[None, None, None, :]
            return np.ascontiguousarray(
                contractions[pair_index[None, :, None, :], r, p]
                + contractions[pair_index[:, None, :, None], s, q]
            )

        def tc_components(context, orbitals, plain, mu, gl_order):
            norb = len(orbitals)
            pairs, pair_index = pair_index_data(norb)
            rmin = X_MIN / mu

            c1, z1 = helper.q1_terms(mu, rmin, gl_order)
            c2, z2 = helper.q2_terms(mu, rmin, gl_order)
            c3, z3 = helper.q3_terms(mu, rmin, gl_order)

            q1_terms = list(zip(c1, z1))
            q2_terms = list(zip(c2, z2))
            q3_terms = list(zip(c3, z3))

            drift_operator = cartesian_convolution(context, q1_terms)
            laplacian_apply = scalar_convolution(context, q2_terms)
            square_operator = cartesian_convolution(context, q3_terms)

            drift_fields = {
                pair: vector_field(
                    drift_operator, plain["rho"][pair], context.prec
                )
                for pair in pairs
            }
            laplacian_fields = {
                pair: laplacian_apply(plain["rho"][pair])
                for pair in pairs
            }
            square_fields = {
                pair: r2_scalar(
                    square_operator, plain["rho"][pair], context.prec
                )
                for pair in pairs
            }

            k1 = drift_tensor(
                orbitals,
                drift_fields,
                pairs,
                pair_index,
                plain["gradients"],
                context.prec,
            )
            k2 = scalar_pair_tensor(
                plain["rho"], laplacian_fields, pairs, pair_index
            )
            k3 = scalar_pair_tensor(
                plain["rho"], square_fields, pairs, pair_index
            )
            return k1, k2, k3
        """
    ),
    md(
        r"""
        ## Hamiltonian assembly and non-Hermitian solve

        In the present physicists' tensor convention,

        $$
        g_{\mathrm{TC}}=g-K_1-K_2-K_3 .
        $$

        PySCF `direct_nosym.contract_2e` does not accept a separate one-body
        contraction plus a raw ERI tensor. The public `absorb_h1e` operation
        must first combine $h$ with the chemists'-ordered two-body tensor.
        `tc_ground_energy` performs this step with a matrix-free
        `LinearOperator` and `eigs(which="SR")`. The plain-limit comparison
        below fixes both the index permutation and the Hamiltonian $1/2$
        convention.
        """
    ),
    code(
        r"""
        results = []
        plain_energies = {}
        for family, data in families.items():
            context = data["context"]
            plain = data["plain"]
            h, g = plain["h"], plain["g"]
            e_plain_tc_path = tc_ground_energy(
                h, g, data["metadata"]["nelec"], context.e_nn
            )
            e_plain_pyscf, _ = fci.direct_spin1.kernel(
                h,
                g.transpose(0, 2, 1, 3),
                h.shape[0],
                data["metadata"]["nelec"],
            )
            e_plain_pyscf += context.e_nn
            assert abs(e_plain_tc_path - e_plain_pyscf) < 1e-10
            plain_energies[family] = float(e_plain_pyscf)
            print(f"{family}: plain-limit energy = {e_plain_pyscf:.10f} Ha")

            for mu in MU_VALUES:
                started = time.time()
                k1, k2, k3 = tc_components(
                    context,
                    data["orbitals"],
                    plain,
                    mu,
                    GL_ORDER,
                )
                g_tc = g - k1 - k2 - k3
                energy = tc_ground_energy(
                    h,
                    g_tc,
                    data["metadata"]["nelec"],
                    context.e_nn,
                )
                assert np.isfinite(energy)
                assert abs(energy - (-2.903724)) < 0.05, (family, mu, energy)
                results.append(
                    {"family": family, "mu": float(mu), "e_tc": float(energy)}
                )
                print(
                    f"{family:9s} mu={mu:3.1f}: E_TC={energy:.10f} Ha "
                    f"({time.time()-started:.1f} s)"
                )

        for family in families:
            at_one = next(
                row["e_tc"]
                for row in results
                if row["family"] == family and row["mu"] == 1.0
            )
            assert at_one < plain_energies[family], (
                family,
                at_one,
                plain_energies[family],
            )
        """
    ),
    code(
        r"""
        payload = {
            "cases": results,
            "plain_energies": plain_energies,
            "exact_reference": -2.903724,
            "gl_order": GL_ORDER,
            "x_min": X_MIN,
            "provenance": provenance,
        }
        output = TUTORIAL / "data" / "expected_results.json"
        with output.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        print("\n mu | E_TC(pure) | E_TC(optimized)")
        print("----+------------+----------------")
        for mu in MU_VALUES:
            pure = next(
                row["e_tc"]
                for row in results
                if row["family"] == "pure" and row["mu"] == mu
            )
            optimized = next(
                row["e_tc"]
                for row in results
                if row["family"] == "optimized" and row["mu"] == mu
            )
            print(f"{mu:3.1f} | {pure: .9f} | {optimized: .9f}")
        print("\nSaved", output.relative_to(TUTORIAL))
        assert len(results) == 2 * len(MU_VALUES)
        """
    ),
    md(
        r"""
        ## Interface boundary

        This tutorial applies the TC convolutions directly to MRA trees. A
        grid-based consumer could instead supply Cartesian points and request
        $(\phi,\partial_x\phi,\partial_y\phi,\partial_z\phi)$ from the same
        trees. That is a compatible numerical interface, but it is a sampling
        adapter rather than the native MRA representation.

        For the reviewed PyTCHInt snapshot
        `4cba43b7387cd3255950e793815f34c354cee926`, the existing MRA side can
        supply (1) the serialized trees plus `world.json`, (2) values and
        Cartesian derivatives at arbitrary requested points, and (3) completed
        one- and two-electron tensors. It cannot hand those trees directly to
        the current PyTCHInt `Basis`: that implementation constructs a PySCF
        molecule, consumes finite `mo_coeff` arrays, builds its own numerical
        grid, and has no public `FunctionTree` reader or orbital-evaluator
        callback. A sampled-grid adapter or Gaussian back-projection is
        possible but controlled and lossy. A lossless Jastrow-optimization
        workflow would therefore add a tree/evaluator adapter at the
        MRA--PyTCHInt boundary and evaluate the MRA orbitals on PyTCHInt's
        requested points.

        The present workflow scans $\mu$; it does not variationally optimize
        the Jastrow factor and it does not feed the TC solution back into the
        orbital optimization.
        """
    ),
]


def main():
    outputs = {
        "part1_mra_dmrg.ipynb": notebook(PART1_CELLS),
        "part2_erfc_tc.ipynb": notebook(PART2_CELLS),
    }
    for name, nb in outputs.items():
        path = TUTORIAL / name
        nbf.write(nb, path)
        print(path)


if __name__ == "__main__":
    main()
