"""Reproducible end-to-end pipeline: generate data -> train -> evaluate -> register.

Run with:  python pipelines/run_pipeline.py [--rows 20000] [--data path.csv]

This is the single entrypoint an orchestrator (Airflow/Prefect/SageMaker
Pipelines) would call as one step; it is intentionally deterministic given the
configured seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import DATA_DIR, ensure_dirs  # noqa: E402
from credit_risk.data import load_dataset, save_dataset  # noqa: E402
from credit_risk.train import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--data", default=None, help="optional CSV path to train on")
    parser.add_argument("--save-data", action="store_true", help="persist generated data")
    args = parser.parse_args()

    ensure_dirs()

    if args.save_data and args.data is None:
        df = load_dataset(n_rows=args.rows)
        out = save_dataset(df, DATA_DIR / "credit_applications.csv")
        print(f"[data] wrote {len(df)} rows -> {out}")

    print("[train] training model...")
    manifest = train(data_path=args.data, n_rows=args.rows)

    print("[metrics]")
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"\n[done] registered model {manifest['model_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
