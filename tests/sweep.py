# Runs the backtest over a list of parameter settings and prints the model's loss for each, to compare ideas.
"""Parameter sweep.

    python3 tests/sweep.py "window_hours=12 damp_grid=0.75" "window_hours=18" ... -- FILE [FILE ...]

Each quoted setting is a space-separated list of key=value overrides of model.PARAMS (see backtest.py --set); an
empty string is the defaults. Both the inside and the final scheme are scored for every setting.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    if "--" not in args:
        raise SystemExit(__doc__)
    split = args.index("--")
    settings, files = args[:split], args[split + 1:]
    for setting in settings:
        params = backtest.parse_sets(setting.split())
        parts = []
        for scheme in ("final", "inside"):
            with redirect_stdout(io.StringIO()):
                summary = backtest.run(files, scheme, params, verbose=False)
            loss, cov = summary["model"]
            parts.append(f"{scheme} {loss:.4f} (cov {cov:.2f})")
        print(f"{setting or 'defaults':<60} " + "  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
