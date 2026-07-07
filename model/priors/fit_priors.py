"""Fit the per-class shape constants used by the geometry pipeline
(MATH.md §4b/§4c) from a Nutrition5k manifest:

  κ  (V = κ·A^{3/2})    log-space intercept: log V − 1.5·log A
  φ  (V = φ·A·h)        median of V / (A·h_max)
  h̄  (flat foods)       median of h_max

Runs on a laptop in seconds; needs the manifest from data/prepare_nutrition5k.py
plus a `class` column (single-ingredient dishes, or region-level rows once the
segmenter lands). Output feeds nutrition/'s shape_priors table and the
DEFAULT_KAPPA fallback in @ppe/pipeline.

    python priors/fit_priors.py --manifest out/n5k-manifest.csv --out out/priors.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="out/priors.json")
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "class" not in df.columns:
        df["class"] = "_global"

    df = df[(df.area_m2 > 1e-4) & (df.volume_m3 > 1e-7) & (df.height_m > 1e-3)]

    priors: dict[str, dict] = {}
    for name, group in df.groupby("class"):
        if len(group) < args.min_samples:
            continue
        log_kappa = (group.volume_m3.map(math.log) - 1.5 * group.area_m2.map(math.log)).median()
        phi = (group.volume_m3 / (group.area_m2 * group.height_m)).median()
        priors[str(name)] = {
            "kappa": round(math.exp(log_kappa), 4),
            "phi": round(min(max(phi, 0.05), 1.0), 3),
            "h_bar_m": round(group.height_m.median(), 4),
            "samples": int(len(group)),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=2))

    if "_global" in priors:
        print(f"global κ = {priors['_global']['kappa']} "
              f"(pipeline DEFAULT_KAPPA placeholder is 0.55 — update it)")
    print(f"{len(priors)} classes → {out}")


if __name__ == "__main__":
    main()
