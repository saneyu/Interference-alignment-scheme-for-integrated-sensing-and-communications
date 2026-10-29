import importlib.util
import json
import csv
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class EntryPointTests(unittest.TestCase):
    def test_changed_sic_mean_is_reported_separately_from_matching_communication(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, run = Path(tmp) / "reference", Path(tmp) / "run"
            for folder in (ref, run):
                (folder / "data").mkdir(parents=True)
                for name in ("communication.csv", "sensing.csv"):
                    row = dict(scenario="BIA", family="lfm", rho="0.3", snr_db="10",
                               format="direct", method="SIC", metric="cee", n="6000", mean="1.75")
                    if name == "sensing.csv" and folder == run:
                        row["mean"] = "1.54"
                    with (folder / "data" / name).open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
            report = reproduce.compare_results(run, ref)
            self.assertEqual(report["status"], "DIFFERENCES")
            self.assertEqual(report["tables"]["communication.csv"]["different_mean_rows"], 0)
            self.assertEqual(report["tables"]["sensing.csv"]["different_mean_rows"], 1)

    def test_reference_and_source_cannot_be_output_targets(self):
        for path in (ROOT, ROOT.parent, ROOT / ".git" / "new_folder", reproduce.EXTENSION, reproduce.REFERENCE,
                     reproduce.EXTENSION / "isac" / "new_folder"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                reproduce.new_output(path)

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "existing.csv"
            marker.write_text("keep this result", encoding="utf-8")
            with self.assertRaises(ValueError):
                reproduce.new_output(root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep this result")

    def test_fixed_figure_grid_is_validated_before_simulation(self):
        source = reproduce.EXTENSION / "configs" / "shared_source.json"
        reproduce.validate_config(source)
        cfg = json.loads(source.read_text(encoding="utf-8"))
        cfg["snr_db"] = [10, 20]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SNR"):
                reproduce.validate_config(path)


if __name__ == "__main__":
    unittest.main()
