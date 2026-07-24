"""Capture portable software revisions for reproducibility. Stdlib only."""
from __future__ import annotations
import glob
import importlib.metadata as md
import json
import os
import sys

_COMMITS = {
    "vampyr": "cfffb56ef83f8850cd4ee83750e41f0fa51ebf0d",
    "mrcpp": "8107aabe28d6e75f04d66c95a94c157731484eae",
    "block2": "a7f7da9274375483ef2a6dcc28bfb50295fdd2db",
    "pytchint": "4cba43b7387cd3255950e793815f34c354cee926",
}


def _conda_meta_version(pkg):
    """Read a release label when a package has no dist-info metadata."""
    prefix = os.path.dirname(os.path.dirname(sys.executable))
    hits = sorted(glob.glob(os.path.join(prefix, "conda-meta", f"{pkg}-*.json")))
    if not hits:
        return None
    base = os.path.basename(hits[-1])[:-5]
    rest = base[len(pkg) + 1:]
    version, sep, _build = rest.rpartition("-")
    return version if sep else None


def _version(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return _conda_meta_version(pkg) or "unknown"


def _commit_entry(package):
    entry = {"commit": _COMMITS[package]}
    version = _version(package)
    if version != "unknown":
        entry["version"] = version
    return entry


def capture():
    software = {
        package: _commit_entry(package)
        for package in ("vampyr", "mrcpp", "block2")
    }
    software["pytchint"] = {
        "commit": _COMMITS["pytchint"],
        "role": "interface reference only",
    }
    for package in (
        "pyscf",
        "numpy",
        "scipy",
        "jupyter",
        "nbformat",
        "ipykernel",
    ):
        software[package] = {"version": _version(package)}
    return {
        "python": sys.version.split()[0],
        "policy": (
            "commit is authoritative when present; "
            "version is the fallback when no commit is available"
        ),
        "software": software,
    }


def write(path):
    p = capture()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(p, f, indent=2, sort_keys=True)
    return p
