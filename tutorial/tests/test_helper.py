import sys, pathlib, math
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import helper

# GL24 is the production cost/accuracy compromise. Measured accuracy at
# (mu=1, x_min=8.7e-5): erfc/r rel ~9.5e-5; erfc^2/4 abs ~1.7e-3 (an x_min
# near-origin-edge artifact, zero-measure for 3D integrals against smooth
# densities). The construction converges to ~1e-8 by GL64, which is where the
# reference engine pins kernel accuracy.

def test_gl24_production_accuracy():
    e = helper.kernel_errors(1.0, 8.7e-5, 24)
    assert e["erfc_over_x_max_rel_err"] < 2e-4, e
    assert e["erfc_sq_over_4_max_err"] < 5e-3, e
    assert e["all_positive"]

def test_gl64_tight_accuracy():
    e = helper.kernel_errors(1.0, 8.7e-5, 64)
    assert e["erfc_over_x_max_rel_err"] < 1e-6, e
    assert e["erfc_sq_over_4_max_err"] < 1e-6, e
    assert e["all_positive"]

def test_mu_universal():
    a = helper.kernel_errors(0.5, 8.7e-5, 24)
    b = helper.kernel_errors(2.0, 8.7e-5, 24)
    assert abs(a["erfc_over_x_max_rel_err"] - b["erfc_over_x_max_rel_err"]) < 1e-6

def test_gl64_tighter_than_gl24():
    e24 = helper.kernel_errors(1.0, 8.7e-5, 24)
    e64 = helper.kernel_errors(1.0, 8.7e-5, 64)
    assert e64["erfc_over_x_max_rel_err"] < e24["erfc_over_x_max_rel_err"]
    assert e64["erfc_sq_over_4_max_err"] < e24["erfc_sq_over_4_max_err"]

def test_q1_reconstructs_at_gl64():
    # q1_terms must reconstruct analytic q1 = erfc(mu r)/(2r) away from r=0.
    mu, n = 1.0, 64
    rmin = 8.7e-5 / mu
    r = np.linspace(0.05, 5.0, 200)
    c, z = helper.q1_terms(mu, rmin, n)
    approx = c @ np.exp(-np.outer(z, r * r))
    assert np.max(np.abs(approx - helper.analytic_kernels(mu, r)["q1"])) < 1e-4

def test_q2_carries_the_negative_gaussian():
    # q2 = erfc/r - (mu/sqrt pi) e^{-mu^2 r^2}: exactly one negative (signed) term,
    # which the Part 2 notebook must sign-split for ConvolutionOperator.
    c, z = helper.q2_terms(1.0, 8.7e-5, 24)
    assert np.sum(np.asarray(c) < 0.0) == 1

if __name__ == "__main__":
    import sys, traceback
    fails = 0
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            try: _f(); print("PASS", _n)
            except Exception: fails += 1; print("FAIL", _n); traceback.print_exc()
    print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(fails)
