/**
 * Builds a small *starter* nutrient bundle of common foods for the demo app, so
 * `apps/demo` shows REAL USDA nutrition (not the single hard-coded mock) without
 * the 181 GB FoodData Central download. Values here are standard USDA reference
 * per-100 g figures (SR Legacy / FNDDS, CC0) for widely-eaten foods — a curated
 * subset, NOT invented: the production bundle is built from the full FDC export
 * (see nutrition/README.md and `npm run etl:bundle`).
 *
 *   node nutrition/starter/build-starter.mjs --out apps/demo/assets/nutrients.sqlite
 *
 * The foods are defined below as plain objects; this script writes them out in
 * FDC CSV shape to a temp dir and runs the SAME buildBundle() the real ETL uses,
 * so the on-device schema is identical. Density is portion-derived where a
 * volumetric measure is given (MATH.md §5); omitted nutrients stay null.
 */
import { mkdtempSync, writeFileSync, readFileSync, copyFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { buildBundle } from "../etl/build-bundle.mjs";
import { FOODS } from "./foods.mjs";

// nutrient_id → USDA definition (only the ones the bundle stores).
const NUTRIENTS = [
  [1008, "Energy", "KCAL"],
  [1003, "Protein", "G"],
  [1005, "Carbohydrate, by difference", "G"],
  [1004, "Total lipid (fat)", "G"],
  [1079, "Fiber, total dietary", "G"],
  [2000, "Sugars, total", "G"],
  [1258, "Fatty acids, total saturated", "G"],
  [1093, "Sodium, Na", "MG"],
  [1253, "Cholesterol", "MG"],
  [1092, "Potassium, K", "MG"],
  [1087, "Calcium, Ca", "MG"],
  [1089, "Iron, Fe", "MG"],
];

const MEASURE_UNITS = [
  [1000, "cup"],
  [1001, "medium"],
];

// Food set (per-100 g USDA reference values + density + shape) is the shared
// source of truth in ./foods.mjs, so the classifier vocabulary, this nutrient
// bundle, and the label→FDC map never drift apart.

const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
const csv = (header, rows) =>
  [header.map(q).join(","), ...rows.map((r) => r.map(q).join(","))].join("\n") + "\n";

function writeFixtures(dir) {
  writeFileSync(join(dir, "measure_unit.csv"), csv(["id", "name"], MEASURE_UNITS));
  writeFileSync(
    join(dir, "nutrient.csv"),
    csv(["id", "name", "unit_name", "nutrient_nbr", "rank"], NUTRIENTS.map(([id, name, unit]) => [id, name, unit, "", ""])),
  );
  writeFileSync(
    join(dir, "food.csv"),
    csv(["fdc_id", "data_type", "description", "food_category_id", "publication_date"],
      FOODS.map((f) => [f.id, f.type, f.desc, "", "2024-10-31"])),
  );
  let fnId = 1;
  const fnRows = [];
  for (const f of FOODS) {
    for (const [nid, amount] of Object.entries(f.n)) {
      fnRows.push([fnId++, f.id, nid, amount, "", "", "", "", "", "", ""]);
    }
  }
  writeFileSync(
    join(dir, "food_nutrient.csv"),
    csv(["id", "fdc_id", "nutrient_id", "amount", "data_points", "derivation_id", "min", "max", "median", "footnote", "min_year_acquired"], fnRows),
  );
  let pId = 1;
  const portionRows = [];
  for (const f of FOODS) {
    if (f.cupG) portionRows.push([pId++, f.id, "1", "1", "1000", "", "", f.cupG, "", "", ""]);
  }
  writeFileSync(
    join(dir, "food_portion.csv"),
    csv(["id", "fdc_id", "seq_num", "amount", "measure_unit_id", "portion_description", "modifier", "gram_weight", "data_points", "footnote", "min_year_acquired"], portionRows),
  );
}

// Per-food shape class (MATH.md §4): most foods pile into a mound; slices and
// fillets lie flat. Each food declares its class in foods.mjs; the store
// resolves each food to its own prior.
const foodClasses = Object.fromEntries(FOODS.map((f) => [f.desc, f.shape]));

const { values } = parseArgs({ options: { out: { type: "string", default: "apps/demo/assets/nutrients.sqlite" }, priors: { type: "string" } } });
const dir = mkdtempSync(join(tmpdir(), "ppe-starter-"));
writeFixtures(dir);
const priors = values.priors ? JSON.parse(readFileSync(values.priors, "utf8")) : null;
const stats = buildBundle({ fdcDir: dir, out: values.out, priors, foodClasses });

// Ship the canonical label→FDC map next to the DB so the demo bundles a local copy.
copyFileSync(new URL("../label-map.json", import.meta.url), join(dirname(values.out), "label-map.json"));

console.log(`starter bundle → ${values.out}: ${stats.foods} foods, ${stats.withDensity} with density, ${stats.shapePriors} shape priors, fts=${stats.fts}; label-map copied`);
