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
});
