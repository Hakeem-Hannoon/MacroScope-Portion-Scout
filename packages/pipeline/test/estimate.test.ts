import { describe, expect, it } from "vitest";
import { ZodError } from "zod";
import {
  type Intrinsics,
  type Vec2,
  type Vec3,
  poseFromArkitCameraToWorld,
  projectPoint,
  worldToCamera,
} from "@ppe/geometry";
import {
  type CapturePayload,
  FixedClassifier,
  FixedSegmenter,
  InMemoryNutrientStore,
  estimateMeal,
} from "../src/index.js";

/**
 * Synthetic nadir capture: ARKit camera 45 cm straight above the table,
 * a 10×10 cm food footprint at the origin, a 20 cm ruler stroke.
 * The camera matrix is serialized exactly as the native module would.
 */
const ARKIT_NADIR_ROW_MAJOR = [1, 0, 0, 0, 0, 0, 1, 0.45, 0, -1, 0, 0, 0, 0, 0, 1];
const K: Intrinsics = { fx: 1500, fy: 1500, cx: 960, cy: 720 };

const pose = poseFromArkitCameraToWorld(ARKIT_NADIR_ROW_MAJOR);
const wtc = worldToCamera(pose);

const squareCorners: Vec3[] = [
  [-0.05, 0, -0.05],
  [0.05, 0, -0.05],
  [0.05, 0, 0.05],
  [-0.05, 0, 0.05],
];
const squarePolygonPx = squareCorners.map((c) => projectPoint(K, wtc, c)!) as Vec2[];

function makePayload(overrides?: Partial<CapturePayload>): CapturePayload {
  return {
    version: 1,
    image: "file:///tmp/capture.heic",
    image_size: [1920, 1440],
    intrinsics: [
      [K.fx, 0, K.cx],
      [0, K.fy, K.cy],
      [0, 0, 1],
    ],
    camera_to_world: ARKIT_NADIR_ROW_MAJOR,
    plane: { normal: [0, 1, 0], d0: 0 },
    strokes: [
      {
        p1: [0.1, 0, -0.05],
        p2: [-0.1, 0, -0.05],
        length_m: 0.2,
        kind: "horizontal",
      },
    ],
    depth: null,
    scale_source: "ruler",
    ...overrides,
  };
}

const riceRecord = {
  label: "white rice, cooked",
  per100: {
    kcal: 130,
    proteinG: 2.7,
    carbsG: 28,
    fatG: 0.3,
    micros: { potassiumMg: 35, sodiumMg: 1 },
  },
  densityGPerMl: 0.67,
  shape: { kind: "mound" as const, kappa: 0.55 },
};

const deps = {
  segmenter: new FixedSegmenter([{ polygonPx: squarePolygonPx.map((p) => [p[0], p[1]] as [number, number]) }]),
  classifier: new FixedClassifier({ label: "white rice, cooked", confidence: 0.86 }),
  nutrients: new InMemoryNutrientStore([riceRecord]),
};

describe("estimateMeal", () => {
  it("produces a grounded, self-consistent estimate from a ruler capture", async () => {
    const result = await estimateMeal(makePayload(), deps);

    expect(result.items).toHaveLength(1);
    const item = result.items[0]!;
    expect(item.label).toBe("white rice, cooked");
    expect(item.geometry.method).toBe("shape_prior");

    // True footprint is 100 cm²; the elevation correction shrinks it a little.
    expect(item.geometry.area_cm2).toBeGreaterThan(75);
    expect(item.geometry.area_cm2).toBeLessThanOrEqual(100);

    // Mass, energy, and macros scale together from per-100g values.
    expect(item.mass_g).not.toBeNull();
    expect(item.kcal).not.toBeNull();
    expect(item.kcal!).toBeCloseTo((item.mass_g! / 100) * 130, -1);
    expect(item.micros?.potassiumMg).toBeCloseTo((item.mass_g! / 100) * 35, -1);
    expect(item.flags).not.toContain("atwater_mismatch");

    // Totals mirror the single item.
    expect(result.totals.kcal).toBe(item.kcal);
    expect(result.totals.protein_g).toBeCloseTo(item.protein_g!, 6);

    // Quality: perfect synthetic calibration, documented error budget.
    expect(result.quality.scale_source).toBe("ruler");
    expect(result.quality.ruler_residual_mm).not.toBeNull();
    expect(result.quality.ruler_residual_mm!).toBeLessThan(0.01);
    expect(result.quality.est_relative_error).toBeCloseTo(0.306, 2);
    expect(result.quality.camera_height_m).toBeCloseTo(0.45, 3);
  });

  it("uses a vertical stroke as measured height and tightens the budget", async () => {
    const payload = makePayload({
      strokes: [
        { p1: [0.1, 0, -0.05], p2: [-0.1, 0, -0.05], length_m: 0.2, kind: "horizontal" },
        { p1: [0.05, 0, 0.05], p2: [0.05, 0.04, 0.05], length_m: 0.04, kind: "vertical" },
      ],
    });
    const result = await estimateMeal(payload, deps);
    const item = result.items[0]!;
    expect(item.geometry.method).toBe("area_height");
    expect(item.geometry.height_cm).toBeCloseTo(4, 5);
    expect(result.quality.est_relative_error).toBeCloseTo(0.207, 2);
  });

  it("refuses to invent nutrition for an unknown food", async () => {
    const result = await estimateMeal(makePayload(), {
      ...deps,
      classifier: new FixedClassifier({ label: "mystery casserole", confidence: 0.42 }),
    });
    const item = result.items[0]!;
    expect(item.flags).toContain("no_db_match");
    expect(item.flags).toContain("low_confidence");
    expect(item.mass_g).toBeNull();
    expect(item.kcal).toBeNull();
    expect(result.totals.kcal).toBe(0);
    // Geometry still reports, so the UI can ask the user with the size known.
    expect(item.geometry.volume_ml).toBeGreaterThan(0);
  });

  it("rejects a malformed payload at the boundary", async () => {
    await expect(
      estimateMeal({ version: 2, image: "x" }, deps),
    ).rejects.toBeInstanceOf(ZodError);
  });
});
