# Helium MRA-DMRG + erfc-TC Tutorial

These two executed notebooks accompany
`MRA_DMRG_TC_Implementation_Note.pdf`:

1. `part1_mra_dmrg.ipynb` projects the He/cc-pVDZ orbitals into MRA,
   validates the plain integrals, optimizes the orbitals with block2, and
   saves the pure and optimized MO trees.
2. `part2_erfc_tc.ipynb` reloads those trees, assembles the GL24 erfc-TC
   operators, and evaluates `mu = 0.5, 1.0` for both orbital families.

## Reproducibility requirements

Create any Python 3.12 environment with the software below and select it as
the notebook's Python kernel. When a commit is listed, commit is authoritative;
the version is only an informational label.

| Software | Required revision |
|---|---|
| VAMPyR | `cfffb56ef83f8850cd4ee83750e41f0fa51ebf0d` (version label `1.0rc1`) |
| MRCPP | `8107aabe28d6e75f04d66c95a94c157731484eae` |
| block2/pyblock2 | `a7f7da9274375483ef2a6dcc28bfb50295fdd2db` (version label `0.5.3`) |
| PySCF | `2.13.1` |
| NumPy | `2.4.6` |
| SciPy | `1.17.1` |
| Jupyter | `1.1.1` |
| nbformat | `5.10.4` |
| ipykernel | `7.2.0` |

PyTCHInt is not a runtime dependency of these notebooks. The interface
discussion uses snapshot
`4cba43b7387cd3255950e793815f34c354cee926`.

Launch Jupyter with:

```bash
jupyter lab tutorial/
```

Run Part I before Part II. Part I takes about two minutes and the default
four-case Part II demo about six minutes on the reference calculation.
Part II can then be rerun directly from the saved checkpoints.

## Outputs

- `data/he_ccpvdz_pure/`: projected and orthonormalized MO trees
- `data/he_ccpvdz_optimized/`: optimized MO trees
- `data/expected_results.json`: executed TC energies
- `data/provenance.json`: portable software revisions

Each checkpoint includes `world.json`, which reconstructs the compatible MRA
world before its `FunctionTree` files are loaded. An MRA tree is an adaptive
coefficient representation, not a fixed Cartesian grid; the detailed
mathematics and operator flow are given in the accompanying PDF and notebook
Markdown.
