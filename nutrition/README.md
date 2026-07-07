# nutrition/

Data ETL: the on-device nutrient + density bundle.

Sources (all storable — licenses checked):

- **USDA FoodData Central** (CC0) — per-100 g energy/macros/micros. Generic foods only for the bundle: SR Legacy + Foundation + FNDDS ≈ 15–17k rows.
- **FNDDS `food_portion`** — 35k+ household-measure gram weights; volumetric measures ("1 cup = 158 g") double as **density** (ρ = g / 236.59 mL).
- **FAO/INFOODS Density Database** — direct density values where FNDDS is silent.

## Output

A versioned SQLite file (~15–30 MB) shipped in the app:

- `foods(fdc_id, name, class, kcal100, protein100, carbs100, fat100, micros…)` + FTS5 index on `name`
- `densities(class, rho_g_per_ml, source)`
- `shape_priors(class, kappa, phi, h_bar)` ← fitted in `model/`, joined here

## Tasks

- [ ] Download + parse FDC CSVs (Foundation, SR Legacy, FNDDS incl. `food_portion`)
- [ ] Derive densities from volumetric portions; merge FAO/INFOODS; manual review of the top ~200 classes
- [ ] Class taxonomy: map classifier labels → food classes → FDC rows (the label↔row join table is the quality-critical artifact)
- [ ] Build + version the SQLite bundle; unit test: every classifier label resolves to nutrients + a density
