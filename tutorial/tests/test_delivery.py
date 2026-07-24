import json
import pathlib


TUTORIAL = pathlib.Path(__file__).resolve().parents[1]

COMMITS = {
    "vampyr": "cfffb56ef83f8850cd4ee83750e41f0fa51ebf0d",
    "mrcpp": "8107aabe28d6e75f04d66c95a94c157731484eae",
    "block2": "a7f7da9274375483ef2a6dcc28bfb50295fdd2db",
    "pytchint": "4cba43b7387cd3255950e793815f34c354cee926",
}


def _contains(path, needle):
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            data = overlap + chunk
            if needle in data:
                return True
            overlap = data[-max(len(needle) - 1, 0):]
    return False


def test_delivery_has_no_local_identifiers():
    forbidden = {
        "macOS home path": b"/" + b"Users" + b"/",
        "Linux home path": b"/" + b"home" + b"/",
        "originating user": b"hong" + b"56",
        "local environment": b"QChem" + b"TNPy312",
        "private temporary path": b"/" + b"private" + b"/",
        "local workspace path": b"Claude" + b"/Projects",
        "file URI": b"file" + b"://",
        "embedded OpenMP workaround": b"KMP_" + b"DUPLICATE_LIB_OK",
    }
    failures = []
    for path in sorted(p for p in TUTORIAL.rglob("*") if p.is_file()):
        for label, needle in forbidden.items():
            if _contains(path, needle):
                failures.append((str(path.relative_to(TUTORIAL)), label))
    assert not failures, failures


def test_delivery_contains_no_transient_artifacts():
    forbidden_names = {".DS_Store", "__pycache__", "tmp_block2"}
    failures = [
        str(path.relative_to(TUTORIAL))
        for path in TUTORIAL.rglob("*")
        if path.name in forbidden_names or path.suffix == ".pyc"
    ]
    assert not failures, failures


def test_readme_documents_authoritative_revisions():
    readme = (TUTORIAL / "README.md").read_text()
    for package, commit in COMMITS.items():
        assert package.lower() in readme.lower()
        assert commit in readme
    assert "commit is authoritative" in readme
    assert "runtime dependency" in readme


def test_result_provenance_matches_sanitized_manifest():
    provenance = json.loads((TUTORIAL / "data" / "provenance.json").read_text())
    results = json.loads((TUTORIAL / "data" / "expected_results.json").read_text())
    assert results["provenance"] == provenance
    assert "platform" not in provenance
    assert "conda_build_strings" not in provenance


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
