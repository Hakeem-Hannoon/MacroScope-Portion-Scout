export interface BundleStats {
  foods: number;
  withDensity: number;
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

export function buildBundle(options: {
  fdcDir: string;
  out: string;
  dataTypes?: string[];
}): BundleStats;

export function openBundle(path: string): {
  count(): number;
  get(fdcId: number): FoodRow | null;
  search(term: string, limit?: number): FoodRow[];
  close(): void;
};
