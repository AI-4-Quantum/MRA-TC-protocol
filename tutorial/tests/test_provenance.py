import sys, pathlib, json, os, re, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import env_provenance

COMMITS = {
    "vampyr": "cfffb56ef83f8850cd4ee83750e41f0fa51ebf0d",
    "mrcpp": "8107aabe28d6e75f04d66c95a94c157731484eae",
    "block2": "a7f7da9274375483ef2a6dcc28bfb50295fdd2db",
    "pytchint": "4cba43b7387cd3255950e793815f34c354cee926",
}

def test_capture_uses_authoritative_commits_and_version_fallbacks():
    p = env_provenance.capture()
    assert set(p) == {"python", "policy", "software"}
    assert "commit is authoritative" in p["policy"]
    assert "platform" not in p
    assert "conda_build_strings" not in p
    for package, commit in COMMITS.items():
        assert p["software"][package]["commit"] == commit
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert p["software"]["pytchint"]["role"] == "interface reference only"
    for package in ("pyscf", "numpy", "scipy", "jupyter", "nbformat"):
        assert p["software"][package]["version"] != "unknown"
        assert "commit" not in p["software"][package]

def test_write_creates_valid_provenance_json():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "provenance.json")
        env_provenance.write(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "python" in data
        assert "policy" in data
        assert "software" in data

if __name__ == "__main__":
    import sys, traceback
    fails = 0
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            try: _f(); print("PASS", _n)
            except Exception: fails += 1; print("FAIL", _n); traceback.print_exc()
    print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(fails)
