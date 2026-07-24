import json
import pathlib


TUTORIAL = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    path = TUTORIAL / name
    assert path.exists(), f"missing tutorial notebook: {path.name}"
    with path.open() as handle:
        notebook = json.load(handle)
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    return notebook


def _source(notebook, cell_type=None):
    cells = notebook["cells"]
    if cell_type is not None:
        cells = [cell for cell in cells if cell["cell_type"] == cell_type]
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def test_part1_has_mra_dmrg_workflow():
    notebook = _load("part1_mra_dmrg.ipynb")
    source = _source(notebook)
    for required in (
        "MultiResolutionAnalysis",
        "ScalingProjector",
        "PoissonOperator",
        "DMRGDriver",
        "get_qc_mpo",
        "saveTree",
        "world.json",
    ):
        assert required in source, required


def test_part1_projection_is_explicit_analytic_lcao():
    source = _source(_load("part1_mra_dmrg.ipynb"))
    for required in (
        "angular_factor",
        "primitive_trees",
        "ao_trees",
        "mo_trees",
        "mo_coefficients[ao, orbital]",
    ):
        assert required in source, required
    for forbidden in ("rng.uniform", "np.dot(reconstructed, reference)"):
        assert forbidden not in source, forbidden


def test_part2_has_tc_operator_workflow():
    notebook = _load("part2_erfc_tc.ipynb")
    source = _source(notebook)
    for required in (
        "loadTree",
        "CartesianConvolution",
        "setCartesianComponents",
        "q1_terms",
        "q2_terms",
        "q3_terms",
        "tc_ground_energy",
        "expected_results.json",
    ):
        assert required in source, required


def test_part2_demo_uses_short_mu_scan_with_full_scan_visible():
    source = _source(_load("part2_erfc_tc.ipynb"))
    expected = """MU_VALUES = (
    0.5,
    # 0.7,
    1.0,
    # 1.5,
    # 2.0,
)"""
    assert expected in source
    assert "assert len(results) == 2 * len(MU_VALUES)" in source


def test_tutorial_explains_tree_and_kernel_approximation():
    combined = _source(_load("part1_mra_dmrg.ipynb"), "markdown")
    combined += "\n" + _source(_load("part2_erfc_tc.ipynb"), "markdown")
    assert "not a grid-value table" in combined
    assert "exact" in combined
    assert "GL24" in combined
    assert "GL32" in combined and "GL64" in combined
    assert "approximation" in combined
    assert "4cba43b7387cd3255950e793815f34c354cee926" in combined
    assert "no public `FunctionTree` reader" in combined


def test_markdown_uses_dollar_display_math_delimiters():
    for name in ("part1_mra_dmrg.ipynb", "part2_erfc_tc.ipynb"):
        markdown = _source(_load(name), "markdown")
        assert r"\[" not in markdown, name
        assert r"\]" not in markdown, name
        assert r"\(" not in markdown, name
        assert r"\)" not in markdown, name


def test_delivered_notebooks_do_not_import_repo_packages():
    for name in ("part1_mra_dmrg.ipynb", "part2_erfc_tc.ipynb"):
        source = _source(_load(name))
        for forbidden in ("from mratc", "import mratc", "mra-dmrg", "experiments.he_tc"):
            assert forbidden not in source


def test_notebooks_use_generic_python_kernel():
    expected = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for name in ("part1_mra_dmrg.ipynb", "part2_erfc_tc.ipynb"):
        notebook = _load(name)
        assert notebook["metadata"]["kernelspec"] == expected, name


def test_executed_notebooks_have_no_error_outputs():
    for name in ("part1_mra_dmrg.ipynb", "part2_erfc_tc.ipynb"):
        notebook = _load(name)
        code_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip()
        ]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, errors


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
