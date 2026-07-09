import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { buildBundle, openBundle } from "../etl/build-bundle.mjs";

const fixtures = fileURLToPath(new URL("../fixtures", import.meta.url));
const workDir = mkdtempSync(join(tmpdir(), "ppe-etl-"));
const bundlePath = join(workDir, "bundle.sqlite");

afterAll(() => rmSync(workDir, { recursive: true, force: true }));

describe("nutrient bundle ETL", () => {
  it("builds a bundle from FDC CSVs, filtering to generic data types", () => {
    const stats = buildBundle({ fdcDir: fixtures, out: bundlePath });
    // The branded soda row is excluded; rice and banana stay.
    expect(stats.foods).toBe(2);
    // Rice has two volumetric (cup) portions; the banana portion ("medium")
    // carries no volume, so only rice gets a density.
    expect(stats.withDensity).toBe(1);
    // No priors passed → a single default _global shape prior is seeded.
    expect(stats.shapePriors).toBe(1);
  });

  it("derives density from volumetric portion weights (MATH.md §5)", () => {
    const bundle = openBundle(bundlePath);
    const rice = bundle.get(1001);
    expect(rice).not.toBeNull();
    // 158 g per 236.588 mL cup → 0.668 g/mL.
    expect(rice!.density_g_per_ml).toBeCloseTo(0.668, 2);
    expect(rice!.density_source).toBe("fdc_portion");
    expect(rice!.kcal100).toBe(130);
    const banana = bundle.get(1002);
    expect(banana!.density_g_per_ml).toBeNull();
    expect(banana!.potassium100).toBe(358);
    bundle.close();
  });

  it("finds foods by text search", () => {
    const bundle = openBundle(bundlePath);
    const hits = bundle.search("rice");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]!.fdc_id).toBe(1001);
    expect(bundle.count()).toBe(2);
    bundle.close();
  });

  it("seeds a default _global shape prior matching the pipeline default (MATH.md §4)", () => {
    const bundle = openBundle(bundlePath); // built above with no priors
    const g = bundle.shapePrior("_global");
    expect(g).not.toBeNull();
    expect(g!.kind).toBe("mound");
    expect(g!.kappa).toBe(0.1687); // == DEFAULT_KAPPA in @ppe/pipeline (Nutrition5k fit)
    expect(g!.phi).toBe(0.446); // == DEFAULT_MOUND_PHI
    expect(g!.h_bar_m).toBe(0.0979);
    expect(g!.source).toBe("default");
    bundle.close();
  });

  it("ingests fitted priors.json into the shape_priors table", () => {
    const out = join(workDir, "with-priors.sqlite");
    const stats = buildBundle({
      fdcDir: fixtures,
      out,
      // the shape of model/priors/fit_priors.py output
      priors: {
        _global: { kappa: 0.42, phi: 0.61, h_bar_m: 0.018, samples: 3200 },
        rice: { kappa: 0.5, phi: 0.7, h_bar_m: 0.02, samples: 120 },
      },
    });
    // both supplied classes; the given _global replaces the default (no extra row)
    expect(stats.shapePriors).toBe(2);
    const bundle = openBundle(out);
    expect(bundle.shapePrior("_global")!.kappa).toBe(0.42);
    expect(bundle.shapePrior("_global")!.source).toBe("nutrition5k_fit");
    expect(bundle.shapePrior("rice")!.phi).toBe(0.7);
    bundle.close();
  });
});
