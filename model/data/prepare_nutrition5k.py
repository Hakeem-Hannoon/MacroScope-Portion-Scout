"""Extract a training manifest from Nutrition5k overhead RGB-D captures.

For every dish with an overhead frame this computes, from the depth map, the
same quantities the phone measures at capture time — so the regressor trains
on exactly the features it will see in production:

  1. fit the table plane from border pixels (least squares — MATH.md §2.2),
  2. height field = plane depth − pixel depth (MATH.md §4a),
  3. food mask = height > threshold,
  4. metric area, max/mean height, integrated volume.

Dataset facts (verified, docs/MODELS.md §6): gs://nutrition5k_dataset, 181.4 GB
full; raw depth is 16-bit with 10,000 units per meter; per-dish mass/kcal in
dish_metadata_cafe*.csv; official splits in dish_ids/splits/. License CC BY 4.0.

    gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" data/n5k/
    gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/metadata" data/n5k/
    gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids" data/n5k/
    python data/prepare_nutrition5k.py --root data/n5k --out out/n5k-manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_UNITS_PER_M = 10_000.0
# Intel RealSense D435 overhead rig. TODO: replace with the per-camera
# calibration from the dataset release if/when located; these are the
# published-spec approximations and cancel out of relative comparisons.
FX = 615.0
FY = 615.0
FOOD_HEIGHT_THRESHOLD_M = 0.005
BORDER_PX = 40


def analyze_depth(depth_path: Path):
    """Plane fit + height field + metric area/volume for one overhead frame."""
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw / DEPTH_UNITS_PER_M
    h, w = depth_m.shape
    valid = depth_m > 0.05

    # 1. Table plane from the border ring: depth(x, y) ≈ ax + by + c.
    ys, xs = np.mgrid[0:h, 0:w]
    border = np.zeros_like(valid)
    border[:BORDER_PX, :] = border[-BORDER_PX:, :] = True
    border[:, :BORDER_PX] = border[:, -BORDER_PX:] = True
    ring = border & valid
    if ring.sum() < 500:
        return None
    A = np.stack([xs[ring], ys[ring], np.ones(ring.sum())], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, depth_m[ring], rcond=None)
    plane_depth = coeffs[0] * xs + coeffs[1] * ys + coeffs[2]

    # 2–3. Height above the table; the food is whatever rises out of it.
    height = np.where(valid, plane_depth - depth_m, 0.0)
    mask = height > FOOD_HEIGHT_THRESHOLD_M
    if mask.sum() < 200:
        return None

    # 4. Per-pixel metric footprint at the table's depth: (Z/fx)·(Z/fy).
    z_table = float(np.median(plane_depth[ring]))
    cell_area_m2 = (z_table / FX) * (z_table / FY)
    area_m2 = float(mask.sum() * cell_area_m2)
    volume_m3 = float(height[mask].sum() * cell_area_m2)
    return {
        "area_m2": area_m2,
        "volume_m3": volume_m3,
        "height_m": float(height[mask].max()),
        "mean_height_m": float(height[mask].mean()),
    }


def load_dish_metadata(metadata_dir: Path) -> dict[str, dict]:
    """dish_id → {mass_g, kcal} from the cafe metadata CSVs."""
    dishes: dict[str, dict] = {}
    for path in sorted(metadata_dir.glob("dish_metadata_cafe*.csv")):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                # Layout: dish_id, total_calories, total_mass, fat, carb, protein, [ingredients…]
                if len(row) < 6 or not row[0].startswith("dish_"):
                    continue
                dishes[row[0]] = {"kcal": float(row[1]), "mass_g": float(row[2])}
    return dishes


def load_splits(split_dir: Path) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for split in ("train", "test"):
        path = split_dir / f"depth_{split}_ids.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    assignment[line.strip()] = split
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="nutrition5k download root")
    parser.add_argument("--out", default="out/n5k-manifest.csv")
    args = parser.parse_args()

    # Load the labels once (dish_id → mass/kcal) and the official split
    # assignment, then walk every overhead capture. `overhead` holds ~5k
    # per-dish folders: iterate it on LOCAL disk — over Drive's FUSE mount this
    # listing plus the per-dish reads below abort (Errno 103).
    root = Path(args.root)
    dishes = load_dish_metadata(root / "metadata")
    splits = load_splits(root / "dish_ids" / "splits")
    overhead = root / "imagery" / "realsense_overhead"

    rows = []
    skipped = 0
    for dish_dir in sorted(overhead.iterdir()):
        dish_id = dish_dir.name
        meta = dishes.get(dish_id)
        rgb = dish_dir / "rgb.png"
        depth = dish_dir / "depth_raw.png"
        # Skip any dish missing a label or either capture — nothing to train on.
        if meta is None or not rgb.exists() or not depth.exists():
            skipped += 1
            continue
        # Depth map → metric geometry (plane fit → height field → area/volume).
        # None means the fit failed (too few valid pixels); also drop trivially
        # light dishes, whose mass labels are unreliable.
        geometry = analyze_depth(depth)
        if geometry is None or meta["mass_g"] <= 1:
            skipped += 1
            continue
        # One manifest row = one training example. image_path is absolute so it
        # resolves wherever the dataset was staged; scale_source is "lidar"
        # because these are RealSense depth captures (the app tags its own rows).
        rows.append(
            {
                "dish_id": dish_id,
                "image_path": str(rgb),
                **geometry,
                "mass_g": meta["mass_g"],
                "kcal": meta["kcal"],
                "scale_source": "lidar",
                "split": splits.get(dish_id, "train"),
            }
        )

    # Write the manifest — the one CSV every downstream step reads (fit_priors.py
    # and mass_regressor_nutrition5k.py). Field names come from the first row, so
    # every row must carry the same keys.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} dishes → {out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
