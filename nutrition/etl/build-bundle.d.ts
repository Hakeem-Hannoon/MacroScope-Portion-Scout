export interface BundleStats {
  foods: number;
  withDensity: number;
  shapePriors: number;
  fts: boolean;
}

export interface FoodRow {
  fdc_id: number;
  description: string;
  data_type: string;
  kcal100: number | null;
  protein100: number | null;
  carbs100: number | null;
  fat100: number | null;
  fiber100: number | null;
  sugar100: number | null;
  satfat100: number | null;
  sodium100: number | null;
  cholesterol100: number | null;
  potassium100: number | null;
  calcium100: number | null;
  iron100: number | null;
  density_g_per_ml: number | null;
  density_source: string | null;
}

/** One row of the shape_priors table (κ/φ/h̄ per food class, MATH.md §4). */
export interface ShapePriorRow {
  class: string;
  kind: string;
  kappa: number | null;
  phi: number | null;
  h_bar_m: number | null;
  samples: number | null;
  source: string;
}

/** Per-class shape priors, keyed by class — the fit_priors.py (priors.json) shape. */
export type PriorsInput = Record<
  string,
  { kind?: string; kappa?: number; phi?: number; h_bar_m?: number; samples?: number }
>;

export function buildBundle(options: {
  fdcDir: string;
  out: string;
  dataTypes?: string[];
  priors?: PriorsInput | null;
}): BundleStats;

export function openBundle(path: string): {
  count(): number;
  get(fdcId: number): FoodRow | null;
  getByDescription(description: string): FoodRow | null;
  search(term: string, limit?: number): FoodRow[];
  shapePrior(className: string): ShapePriorRow | null;
  close(): void;
};
