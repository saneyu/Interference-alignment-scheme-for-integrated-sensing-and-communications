"""Portable entry point for the archived BIA/TIM shared-source extension."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXTENSION = ROOT / "extensions" / "shared_source"
REFERENCE = EXTENSION / "reference"


def run(arguments: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Avoid oversubscribing the many small linear algebra operations.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print("Running:", " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, *map(str, arguments)], cwd=EXTENSION,
                   env=env, check=True)


def verify(out: Path, write_report: bool = False) -> None:
    args = [str(EXTENSION / "verify_shared_source.py"), str(out)]
    if write_report:
        args.append("--write-report")
    run(args)


def validate_config(path: Path) -> None:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    # The archived plotting code has fixed panels at these grid points.
    if cfg.get("families") != ["dft", "lfm", "walsh", "qpsk"]:
        raise ValueError("The 12-figure pipeline requires families = dft, lfm, walsh, qpsk in this order.")
    if not {0, 10, 20, 30, 35}.issubset(cfg.get("snr_db", [])):
        raise ValueError("The 12-figure pipeline requires SNR points 0, 10, 20, 30, 35 dB.")
    rhos = cfg.get("sensing_fractions", [])
    if not rhos or cfg.get("main_sensing_fraction") not in rhos:
        raise ValueError("main_sensing_fraction must occur in sensing_fractions.")
    if any(not 0 < r < 1 for r in rhos):
        raise ValueError("Sensing fractions must be strictly between 0 and 1.")
    for field in ("communication_trials", "sensing_trials", "blocks", "likelihood_batch", "sic_iterations"):
        value = cfg.get(field)
        minimum = 2 if field in ("communication_trials", "sensing_trials", "blocks") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}.")
    if cfg.get("peak_expected_power_cap") != 1.0:
        raise ValueError("The archived verifier expects peak_expected_power_cap = 1.0.")


def new_output(path: Path) -> Path:
    out = path.resolve()
    # Never allow a run to overwrite code or the reference snapshot.
    git_dir = ROOT / ".git"
    if out == ROOT or out in ROOT.parents or out == EXTENSION or EXTENSION in out.parents or out == git_dir or git_dir in out.parents:
        raise ValueError("Choose an output directory outside extensions/shared_source (for example results/my_run).")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Output already exists and is not empty: {out}. Choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def record_environment(out: Path, mode: str) -> None:
    data = {
        "mode": mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        "note": "Monte Carlo source hashes and seeds are in manifest.json and config_used.json.",
    }
    (out / "run_environment.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def compare_results(out: Path, reference: Path = REFERENCE) -> dict:
    """Compare primary statistical means; internal validation is a separate check."""
    result = {"status": "MATCH", "relative_tolerance": 1e-9, "absolute_tolerance": 1e-11,
              "tables": {}, "difference_examples": []}
    for name in ("communication.csv", "sensing.csv"):
        tables = []
        for folder in (reference, out):
            with (folder / "data" / name).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            keys = ("scenario", "family", "rho", "snr_db", "format", "method", "metric")
            indexed = {tuple(row.get(key, "") for key in keys): row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f"Duplicate statistical rows in {folder / 'data' / name}")
            tables.append(indexed)
        expected, actual = tables
        if expected.keys() != actual.keys():
            raise ValueError(f"Experiment grid differs in {name}; compare equivalent configurations.")
        changed = 0
        groups = Counter()
        for key, before in expected.items():
            after = actual[key]
            left, right = float(before["mean"]), float(after["mean"])
            match = (before["n"] == after["n"] and math.isfinite(left) and math.isfinite(right)
                     and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-11))
            if not match:
                changed += 1
                group = "/".join(before.get(k, "") for k in ("scenario", "family", "method", "format", "metric"))
                groups[group] += 1
                if len(result["difference_examples"]) < 12:
                    result["difference_examples"].append({"table": name, "case": dict(zip(keys, key)),
                        "reference_mean": left, "new_mean": right, "reference_n": int(before["n"]), "new_n": int(after["n"])})
        result["tables"][name] = {"rows": len(expected), "matching_mean_rows": len(expected) - changed,
                                  "different_mean_rows": changed, "difference_groups": dict(groups)}
        if changed:
            result["status"] = "DIFFERENCES"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="Run the shared-source mathematical contract tests.")
    check = sub.add_parser("verify", help="Validate saved samples, statistics, resources, plots and source hashes; writes no report.")
    check.add_argument("--out", type=Path, default=REFERENCE)
    compare = sub.add_parser("compare", help="Compare a full run against archived means; exit 1 on differences.")
    compare.add_argument("--out", type=Path, default=ROOT / "results" / "shared_source_full")
    for name, description in (("quick", "Run reduced Monte Carlo counts; not publication results."),
                              ("full", "Rerun the archived full Monte Carlo configuration.")):
        item = sub.add_parser(name, help=description)
        item.add_argument("--out", type=Path, default=ROOT / "results" / f"shared_source_{name}")
        item.add_argument("--config", type=Path, default=EXTENSION / "configs" / "shared_source.json")
    plot = sub.add_parser("plot", help="Regenerate all 12 figures from a copy of the archived data.")
    plot.add_argument("--out", type=Path, default=ROOT / "results" / "shared_source_replot")
    args = parser.parse_args(argv)
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("Run without -O/-OO or PYTHONOPTIMIZE: the original verifier uses assertions.")
    try:
        if args.command == "test":
            run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_shared_source*.py", "-v"])
            run(["-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"])
        elif args.command == "verify":
            verify(args.out.resolve())
        elif args.command == "compare":
            result = compare_results(args.out.resolve())
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "MATCH" else 1
        elif args.command == "plot":
            verify(REFERENCE)
            out = new_output(args.out)
            shutil.copytree(REFERENCE, out, dirs_exist_ok=True)
            run(["-m", "isac.shared_source_plots", str(out)])
            verify(out, write_report=True)
            record_environment(out, args.command)
        else:
            cfg = args.config.resolve()
            validate_config(cfg)
            out = new_output(args.out)
            command = [str(EXTENSION / "run_shared_source.py"), "--config", str(cfg), "--out", str(out)]
            if args.command == "quick":
                command.append("--quick")
            run(command)
            verify(out, write_report=True)
            record_environment(out, args.command)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
