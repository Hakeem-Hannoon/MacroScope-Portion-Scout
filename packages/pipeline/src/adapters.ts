import type { NutrientsPer100g } from "@ppe/geometry";
import type { CapturePayload } from "./contracts";

/** One segmented food region; polygon in pixel coordinates of the stored image. */
export interface Region {
  polygonPx: [number, number][];
}

/** Segmentation model adapter (SAM-class, SegFormer fine-tune, …). */
export interface Segmenter {
  segment(imageUri: string, imageSize: [number, number]): Promise<Region[]>;
}

export interface ClassifierResult {
  label: string;
  confidence: number;
  topK?: { label: string; confidence: number }[];
}

/** Classification model adapter (fine-tuned classifier or CLIP zero-shot). */
export interface Classifier {
  classify(imageUri: string, region: Region): Promise<ClassifierResult>;
}

export type ShapeKind = "mound" | "flat" | "container";

export interface FoodShape {
  kind: ShapeKind;
  /** V = κ·A^(3/2) shape-prior constant (MATH.md §4c). */
  kappa?: number;
  /** Thickness prior for flat foods, meters. */
  hBarM?: number;
  /** Fill factor for the area×height route (MATH.md §4b). */
  phi?: number;
}

export interface FoodRecord {
  label: string;
  per100: NutrientsPer100g;
  densityGPerMl: number;
  shape: FoodShape;
}

/** Nutrient database adapter (the SQLite bundle on device; a Map in tests). */
export interface NutrientStore {
  lookup(label: string): Promise<FoodRecord | null>;
}

/**
 * Optional depth adapter: decodes the capture's depth map and returns a
 * metric height field over one region (heights above the table plane on a
 * regular metric grid — MATH.md §4a).
 */
export interface DepthProvider {
  heightField(
    payload: CapturePayload,
    region: Region,
  ): Promise<{ heights: ArrayLike<number>; cellAreaM2: number; maxHeightM?: number }>;
}
