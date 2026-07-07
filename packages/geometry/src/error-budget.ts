/** Where the metric scale came from (MATH.md §7, best to worst). */
export type ScaleSource = "lidar" | "ruler" | "reference_object" | "stated" | "none";

/** Independent relative-error terms of the mass estimate (MATH.md §8). */
export interface ErrorSources {
  /** Relative error of the linear scale. Enters the budget doubled (area ∝ s²). */
  scaleRel: number;
  /** Relative area error from the segmentation boundary. */
  segmentationRel: number;
  /** Relative error of the height / shape term. */
  heightRel: number;
  /** Relative error of the density value. */
  densityRel: number;
}

/**
 * Combine independent relative errors in quadrature, MATH.md §8:
 *   (σ_m/m)² ≈ (2·σ_s/s)² + (σ_A/A)²_seg + (σ_h/h)² + (σ_ρ/ρ)²
 */
export function combinedRelativeError(e: ErrorSources): number {
  return Math.hypot(2 * e.scaleRel, e.segmentationRel, e.heightRel, e.densityRel);
}

const SCALE_REL: Record<ScaleSource, number> = {
  lidar: 0.015,
  ruler: 0.025,
  reference_object: 0.045,
  stated: 0.075,
  none: 0.25,
};

/**
 * The default budget for a given capture quality. `heightMeasured` is true
 * when the height came from LiDAR integration or a vertical ruler stroke
 * (as opposed to a per-class shape prior).
 */
export function errorPreset(source: ScaleSource, heightMeasured: boolean): ErrorSources {
  return {
    scaleRel: SCALE_REL[source],
    segmentationRel: 0.08,
    heightRel: heightMeasured ? 0.1 : 0.25,
    densityRel: 0.15,
  };
}
