// These types mirror packages/pipeline (adapters.ts) and packages/geometry
// (energy.ts). They are duplicated here on purpose so `nutrition/` stays
// dependency-free; the store satisfies the pipeline's `NutrientStore` interface
// structurally. Keep in sync if those contracts change.

export type MicroKey =
  | "fiberG"
  | "sugarG"
  | "satFatG"
  | "sodiumMg"
  | "cholesterolMg"
  | "potassiumMg"
  | "calciumMg"
  | "ironMg";

export type Micros = Partial<Record<MicroKey, number>>;

export interface NutrientsPer100g {
  kcal: number;
  proteinG: number;
  carbsG: number;
  fatG: number;
  micros?: Micros;
}

export interface FoodShape {
  kind: "mound" | "flat" | "container";
  kappa?: number;
  hBarM?: number;
  phi?: number;
}

export interface FoodRecord {
  label: string;
  per100: NutrientsPer100g;
  densityGPerMl: number;
  shape: FoodShape;
}

/** Concrete NutrientStore over the SQLite bundle (Node built-in sqlite). */
export interface SqliteNutrientStore {
  lookup(label: string): Promise<FoodRecord | null>;
  close(): void;
}

/**
 * Open a read-only nutrient store over a built bundle.
 * @param aliases curated classifier-label → fdc_id (number) or search-term
 *   (string) map — the label→FDC-row artifact from STATUS.md.
 */
export function openNutrientStore(
  path: string,
  options?: { aliases?: Record<string, number | string> },
): SqliteNutrientStore;
