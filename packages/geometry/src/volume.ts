export const M3_TO_ML = 1e6;

/**
 * Shape-prior volume, MATH.md §4c: V = κ·A^(3/2). Holds when the food is a
 * scaled copy of a canonical per-class shape (linear size ∝ √A, so
 * V ∝ A^(3/2)). κ is fitted per class from Nutrition5k depth data.
 */
export function volumeShapePriorM3(areaM2: number, kappa: number): number {
  if (areaM2 < 0 || kappa < 0) throw new Error("area and kappa must be non-negative");
  return kappa * Math.pow(areaM2, 1.5);
}

/**
 * Area × height volume, MATH.md §4b: V = φ·A·h with a per-class fill factor
 * (slab/cylinder 1, dome 2/3, cone 1/3, typical mound ≈ 0.5–0.6).
 */
export function volumeAreaHeightM3(areaM2: number, heightM: number, phi = 1): number {
  if (areaM2 < 0 || heightM < 0 || phi < 0 || phi > 1) {
    throw new Error("invalid area/height/fill factor");
  }
  return phi * areaM2 * heightM;
}

/**
 * Height-field integration, MATH.md §4a: V = Σ max(0, h)·ΔA over a metric
 * grid of per-cell heights above the table plane (LiDAR / depth route).
 */
export function integrateHeightFieldM3(heights: ArrayLike<number>, cellAreaM2: number): number {
  if (cellAreaM2 <= 0) throw new Error("cell area must be positive");
  let v = 0;
  for (let i = 0; i < heights.length; i++) {
    const h = heights[i]!;
    if (h > 0) v += h;
  }
  return v * cellAreaM2;
}

/** Mass from volume and density (MATH.md §5): m = ρ·V. */
export function massG(volumeMl: number, densityGPerMl: number): number {
  if (volumeMl < 0 || densityGPerMl <= 0) throw new Error("invalid volume or density");
  return volumeMl * densityGPerMl;
}
