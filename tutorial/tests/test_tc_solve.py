import pathlib
import sys

import numpy as np
from pyscf import fci


TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def test_tc_ground_energy_reproduces_hermitian_fci():
    from tc_solve import tc_ground_energy

    h = np.array([[-1.2, 0.05], [0.05, -0.3]])
    eri_chem = np.zeros((2, 2, 2, 2))
    eri_chem[0, 0, 0, 0] = 0.7
    eri_chem[1, 1, 1, 1] = 0.5
    eri_chem[0, 0, 1, 1] = eri_chem[1, 1, 0, 0] = 0.2
    eri_chem[0, 1, 0, 1] = eri_chem[1, 0, 1, 0] = 0.1
    g_phys = eri_chem.transpose(0, 2, 1, 3)

    reference, _ = fci.direct_spin1.kernel(h, eri_chem, 2, 2)
    actual = tc_ground_energy(h, g_phys, 2, 0.0)
    assert abs(actual - reference) < 1e-11, (actual, reference)


def test_tc_ground_energy_accepts_nonhermitian_two_body_tensor():
    from tc_solve import compute_fci_ground_state, tc_ground_energy

    h = np.diag([-1.0, -0.2])
    gt = np.zeros((2, 2, 2, 2))
    gt[0, 0, 0, 0] = 0.6
    gt[0, 1, 1, 0] = 0.03
    gt[1, 0, 0, 1] = -0.01
    eigenvalue, ci_dim, h_op = compute_fci_ground_state(
        h, gt.transpose(0, 2, 1, 3), 2, (1, 1)
    )
    assert ci_dim == 4
    assert h_op.shape == (ci_dim, ci_dim)
    assert h_op.matvec(np.ones(ci_dim)).shape == (ci_dim,)
    assert np.isfinite(eigenvalue)

    energy = tc_ground_energy(h, gt, 2, 0.0)
    assert np.isfinite(energy)


if __name__ == "__main__":
    import traceback

    failures = 0
    for test_name, test_fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(test_fn):
            try:
                test_fn()
                print("PASS", test_name)
            except Exception:
                failures += 1
                print("FAIL", test_name)
                traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    raise SystemExit(failures)
