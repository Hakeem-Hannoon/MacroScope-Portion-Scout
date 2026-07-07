/** Micronutrients carried end-to-end (grams or milligrams as suffixed). */
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

/** Nutrient content per 100 g of the food, as stored in the database. */
export interface NutrientsPer100g {
  kcal: number;
  proteinG: number;
  carbsG: number;
  fatG: number;
  micros?: Micros;
}

export interface NutrientAmounts {
  kcal: number;
  proteinG: number;
  carbsG: number;
  fatG: number;
  micros: Micros;
}

/** Scale per-100 g database values to an estimated mass (MATH.md §6). */
export function nutrientsForMassG(per100: NutrientsPer100g, massG: number): NutrientAmounts {
  if (massG < 0) throw new Error("mass must be non-negative");
  const f = massG / 100;
  const micros: Micros = {};
  for (const [key, value] of Object.entries(per100.micros ?? {})) {
    if (typeof value === "number" && Number.isFinite(value)) {
      micros[key as MicroKey] = value * f;
    }
  }
  return {
    kcal: per100.kcal * f,
    proteinG: per100.proteinG * f,
    carbsG: per100.carbsG * f,
    fatG: per100.fatG * f,
    micros,
  };
}

/** Atwater energy identity: kcal ≈ 4·protein + 4·carbs + 9·fat (MATH.md §6). */
export function atwaterKcal(proteinG: number, carbsG: number, fatG: number): number {
  return 4 * proteinG + 4 * carbsG + 9 * fatG;
}

/**
 * Relative deviation between a stated kcal value and the Atwater estimate.
 * Values above ~0.15 indicate a bad database match or a bad parse.
 * Returns null when kcal is too small to compare.
 */
export function atwaterDeviation(
  kcal: number,
  proteinG: number,
  carbsG: number,
  fatG: number,
): number | null {
  if (kcal < 1) return null;
  return Math.abs(kcal - atwaterKcal(proteinG, carbsG, fatG)) / kcal;
}
