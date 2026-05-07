"""Diagnose how CheMeleon is (or isn't) reachable in this Python env.

Run::

    python -m src.v3.diagnose_chemeleon

It prints a bunch of probes and tells you which install path works for
your particular chemprop version. Then set ``CHEMELEON`` according to
the recommended option and re-run the main pipeline.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _h(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    _h("Python / chemprop / lightning versions")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    for pkg in ("chemprop", "lightning", "torch",
                "chemeleon", "Chemeleon",
                "chemprop_foundation_models"):
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {pkg:30s} {version}    "
                  f"({getattr(mod, '__file__', '?')})")
        except ImportError as e:
            print(f"  {pkg:30s} NOT IMPORTABLE   ({e})")

    _h("Pip-level visibility")
    for name in ("chemprop", "chemeleon", "ChemMeleon", "chem-meleon"):
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "show", name],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                first = r.stdout.splitlines()[:3]
                print(f"  pip show {name}: " + " | ".join(first))
            else:
                print(f"  pip show {name}: not installed")
        except Exception as e:
            print(f"  pip show {name}: {e}")

    _h("Looking for CheMeleon classes in chemprop's namespace")
    try:
        import chemprop
    except ImportError:
        print("  chemprop itself not installed; nothing to probe")
        return

    print(f"  chemprop.__version__ = {getattr(chemprop, '__version__', '?')}")
    print(f"  chemprop installed at {Path(chemprop.__file__).parent}")

    # Common locations across chemprop 2.0 → 2.2
    candidates = [
        ("chemprop.foundation_models",       "ChemeleonFoundation"),
        ("chemprop.foundation_models",       "Chemeleon"),
        ("chemprop.foundation_models",       "CheMeleon"),
        ("chemprop.foundation",              "ChemeleonFoundation"),
        ("chemprop.foundation",              "Chemeleon"),
        ("chemprop.foundations",             "Chemeleon"),
        ("chemprop.featurizers",             "Chemeleon"),
        ("chemprop.featurizers",             "ChemeleonFoundation"),
        ("chemprop.featurizers.foundation",  "Chemeleon"),
        ("chemprop.models.foundation",       "Chemeleon"),
        ("chemprop.cli.utils.foundation",    "Chemeleon"),
    ]
    found_python = []
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            present = cls is not None
        except ImportError:
            present = False
        marker = "✓" if present else "·"
        print(f"  {marker} {mod_name}.{cls_name}")
        if present:
            found_python.append((mod_name, cls_name))

    _h("Looking for `chemeleon` as a standalone Python package")
    for pkg in ("chemeleon", "Chemeleon"):
        try:
            mod = importlib.import_module(pkg)
            print(f"  ✓ import {pkg} OK at {Path(mod.__file__).parent}")
            classes = [a for a in dir(mod) if "MeleonF" in a or "meleon" in a.lower()]
            if classes:
                print(f"    classes seen: {classes}")
        except ImportError as e:
            print(f"  · {pkg}: {e}")

    _h("Looking for chemprop CLI foundation flags")
    chemprop_cli = shutil.which("chemprop")
    print(f"  `chemprop` CLI path: {chemprop_cli}")
    if chemprop_cli:
        try:
            r = subprocess.run([chemprop_cli, "train", "--help"],
                               capture_output=True, text=True, timeout=20)
            help_txt = (r.stdout + r.stderr).lower()
            for keyword in ("--foundation", "--from-foundation",
                            "--from-pretrained", "chemeleon"):
                if keyword in help_txt:
                    # Print the line containing the keyword
                    for line in (r.stdout + r.stderr).splitlines():
                        if keyword in line.lower():
                            print(f"    > {line.strip()}")
                            break
                    else:
                        print(f"    found keyword: {keyword}")
        except Exception as e:
            print(f"    couldn't probe CLI: {e}")

    _h("Looking for any CheMeleon-shaped checkpoint on disk")
    home = Path.home()
    search_dirs = [
        home / ".cache" / "chemprop",
        home / ".cache" / "chemeleon",
        home / ".cache" / "huggingface",
        Path("/tmp"),
        Path("/mnt") / "data",
    ]
    found_ckpt = []
    for d in search_dirs:
        if not d.exists():
            continue
        try:
            for path in d.rglob("*"):
                if path.is_file():
                    name_lower = path.name.lower()
                    if (("chemeleon" in name_lower) or
                        ("foundation" in name_lower and path.suffix
                         in (".pt", ".ckpt", ".pth", ".bin"))):
                        size_mb = path.stat().st_size / (1 << 20)
                        if size_mb > 1:    # skip tiny config files
                            found_ckpt.append(f"  {path} ({size_mb:.1f} MB)")
        except (PermissionError, OSError):
            pass
    if found_ckpt:
        print("  Possible checkpoints found:")
        for l in found_ckpt[:20]:
            print(l)
    else:
        print("  No chemeleon-named checkpoint found in common dirs.")

    _h("Recommendation")
    if found_python:
        m, c = found_python[0]
        print(f"  ✓ Set CHEMELEON=auto and the wrapper will use {m}.{c}")
    elif found_ckpt:
        print(f"  ✓ Set CHEMELEON='{found_ckpt[0].strip().split()[0]}' to load the checkpoint manually")
    else:
        print("  ✗ CheMeleon is NOT installed. Install options to try:")
        print("    1) pip install chemprop --upgrade   "
              "# the latest chemprop bundles foundation models")
        print("    2) pip install chemeleon              "
              "# if a standalone package is published")
        print("    3) Manually download from:")
        print("       https://github.com/aganitha/CheMeleon  (if it exists)")
        print("       and set CHEMELEON=/absolute/path/to/chemeleon.pt")


if __name__ == "__main__":
    main()
