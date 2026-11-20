# Unit tests: output shape and validity, determinism, nulls and missing trophies, tiny histories, speed, scoring.
"""Run from the repository root:  python3 -m unittest discover -s tests -v

Set MAKEFUN_KIT_INPUT to the kit's input/ directory to also check the practice cases."""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest  # noqa: E402
import cases  # noqa: E402
import main  # noqa: E402
import model  # noqa: E402
import quantiles  # noqa: E402


def run_main(case_files: dict, hash_seed: str = "0") -> tuple:
    """Runs main.py as the grader does on the given {name: case}; returns ({name: bytes}, seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in"
        out = Path(tmp) / "out"
        inp.mkdir()
        for name, case in case_files.items():
            (inp / f"{name}.in").write_text(json.dumps(case), encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": hash_seed,
               "MAKEFUN_INPUT_DIR": str(inp), "MAKEFUN_OUTPUT_DIR": str(out)}
        started = time.perf_counter()
        subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")], env=env, check=True, cwd=tmp)
        seconds = time.perf_counter() - started
        return {p.stem: p.read_bytes() for p in sorted(out.glob("*.out"))}, seconds


class Validity(unittest.TestCase):
    def assert_valid(self, forecast: dict, case: dict):
        ids = [t["id"] for t in case["trophies"]]
        self.assertTrue(quantiles.is_valid(forecast, ids))
        self.assertEqual(len(forecast["trophies"]), len(ids))
        for row in forecast["trophies"]:
            q = row["q"]
            self.assertEqual(len(q), 5)
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 100 for v in q))
            self.assertTrue(all(a <= b for a, b in zip(q, q[1:])))
        self.assertLessEqual(len(json.dumps(forecast, separators=(",", ":"))), 65536)

    def test_full_case_has_55_valid_rows(self):
        case = cases.build()
        forecast = model.forecast_case(case)
        self.assert_valid(forecast, case)
        self.assertEqual(sorted(r["id"] for r in forecast["trophies"]), list(range(1, 56)))

    def test_model_output_is_its_own_not_the_fallback(self):
        case = cases.build()
        self.assertEqual(main.forecast(case), model.forecast_case(case))

    def test_nulls_in_reads(self):
        case = cases.build(null_share=0.2)
        self.assert_valid(model.forecast_case(case), case)

    def test_trophy_never_read_and_missing_from_last_read(self):
        case = cases.build()
        for read in case["history"]:
            read["rates"][4] = None
        case["history"][-1]["rates"][7] = None
        self.assert_valid(model.forecast_case(case), case)

    def test_rates_shorter_than_trophies(self):
        case = cases.build()
        for read in case["history"][-20:]:
            read["rates"] = read["rates"][:50]
        self.assert_valid(model.forecast_case(case), case)
        self.assert_valid(main.forecast(case), case)

    def test_few_reads(self):
        for count in (0, 1, 2, 5):
            case = cases.build()
            case["history"] = case["history"][:count]
            self.assert_valid(model.forecast_case(case), case)
            self.assert_valid(main.forecast(case), case)

    def test_rates_at_the_clamps(self):
        case = cases.build()
        for read in case["history"]:
            read["rates"][2] = 0
            read["rates"][3] = 100
            read["rates"][5] = 0.01
            read["rates"][6] = 99.99
        self.assert_valid(model.forecast_case(case), case)

    def test_garbage_values_fall_back_validly(self):
        case = cases.build()
        case["history"][-1]["rates"][9] = "n/a"
        case["history"][-2]["at"] = "not a time"
        self.assert_valid(main.forecast(case), case)

    def test_platinum_never_above_another_trophy(self):
        case = cases.build()
        rows = model.forecast_case(case)["trophies"]
        plat = rows[0]["q"]
        for row in rows[1:]:
            self.assertTrue(all(a <= b for a, b in zip(plat, row["q"])))

    def test_practice_cases_if_present(self):
        kit = os.environ.get("MAKEFUN_KIT_INPUT")
        if not kit:
            self.skipTest("MAKEFUN_KIT_INPUT not set")
        for path in sorted(Path(kit).glob("*.in")):
            case = json.loads(path.read_text(encoding="utf-8"))
            self.assert_valid(model.forecast_case(case), case)


class Determinism(unittest.TestCase):
    def test_two_runs_byte_identical(self):
        files = {"a00": cases.build(seed=1), "a01": cases.build(seed=2, null_share=0.1)}
        first, _ = run_main(files)
        second, _ = run_main(files)
        self.assertEqual(first, second)
        third, _ = run_main(files, hash_seed="12345")
        self.assertEqual(first, third)
        self.assertEqual(sorted(first), ["a00", "a01"])


class Speed(unittest.TestCase):
    def test_36_hours_at_10_minutes_under_2_seconds(self):
        outputs, seconds = run_main({"s00": cases.build(hours=36.0, step_minutes=10.0)})
        self.assertIn("s00", outputs)
        self.assertLess(seconds, 2.0, f"took {seconds:.2f}s")

    def test_every_read_a_new_page_60_hours_under_2_seconds(self):
        outputs, seconds = run_main({"s01": cases.every_read_distinct(60.0)})
        self.assertIn("s01", outputs)
        self.assertLess(seconds, 2.0, f"took {seconds:.2f}s")


class Scoring(unittest.TestCase):
    def test_rules_worked_example(self):
        trophies = [{"id": 1}, {"id": 2}]
        forecast = {"trophies": [{"id": 1, "q": [10, 15, 20, 25, 35]}, {"id": 2, "q": [10, 15, 20, 25, 35]}]}
        loss, items, covered, scored = backtest.score(forecast, [22, 8], trophies)
        self.assertAlmostEqual(loss / items, 0.403404, places=6)
        self.assertEqual((covered, scored), (1, 2))


if __name__ == "__main__":
    unittest.main()
