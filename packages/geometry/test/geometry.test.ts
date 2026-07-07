import { describe, expect, it } from "vitest";
import {
  type CameraPose,
  type Intrinsics,
  type Mat3,
  type Plane,
  type Vec2,
  type Vec3,
  applyHomography,
  atwaterDeviation,
  cameraHeightAbovePlane,
  combinedRelativeError,
  cross,
  dist2d,
  elevationLengthFactor,
  errorPreset,
  integrateHeightFieldM3,
  intersectRayPlane,
  intrinsicsFromMatrix,
  massG,
  mat3FromCols,
  mat3Inverse,
  mat3Mul,
  metricPolygonAreaM2,
  normalize,
  nutrientsForMassG,
  pixelRay,
  planeBasis,
  planeCoords,
  planePoint,
  planeToImageHomography,
  polygonArea,
  poseFromArkitCameraToWorld,
  projectPoint,
  rescaleIntrinsics,
  rulerResidualM,
  sub,
  volumeAreaHeightM3,
  volumeShapePriorM3,
  worldToCamera,
} from "../src/index.js";

/** Build a CV-convention camera pose looking from `center` toward `target`. */
function lookAtPoseCV(center: Vec3, target: Vec3, upHint: Vec3): CameraPose {
  const z = normalize(sub(target, center)); // forward
  const x = normalize(cross(upHint, z));
  const y = cross(z, x);
  return { rotation: mat3FromCols(x, y, z), center };
}

const K: Intrinsics = { fx: 1500, fy: 1500, cx: 960, cy: 720 };
const TABLE: Plane = { n: [0, 1, 0], d0: 0 };

/** Angled shot: camera half a meter up, off to the side, aimed at the origin. */
const angled = lookAtPoseCV([0.06, 0.5, 0.35], [0, 0, 0], [0, 1, 0]);
const angledWtc = worldToCamera(angled);

/** Nadir shot straight down from 45 cm (upHint ⊥ forward to stay non-degenerate). */
const nadir = lookAtPoseCV([0, 0.45, 0], [0, 0, 0], [0, 0, 1]);
const nadirWtc = worldToCamera(nadir);

describe("mat3", () => {
  it("inverse round-trips", () => {
    const m: Mat3 = [2, 0.3, -1, 0.5, 1.8, 0.2, -0.4, 0.1, 1.1];
    const id = mat3Mul(m, mat3Inverse(m));
    const expected = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    id.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 10));
  });
});

describe("camera model (MATH.md §2)", () => {
  it("ray through a projected pixel returns to the same plane point", () => {
    const points: Vec3[] = [
      [0.02, 0, -0.03],
      [-0.08, 0, 0.05],
      [0.1, 0, 0.09],
    ];
    for (const x of points) {
      const px = projectPoint(K, angledWtc, x);
      expect(px).not.toBeNull();
      const ray = pixelRay(K, angled, px!);
      const back = intersectRayPlane(ray.origin, ray.dir, TABLE);
      expect(back).not.toBeNull();
      back!.forEach((v, i) => expect(v).toBeCloseTo(x[i]!, 9));
    }
  });

  it("rescaled intrinsics project consistently (MATH.md §9.1)", () => {
    const x: Vec3 = [0.04, 0, -0.02];
    const full = projectPoint(K, angledWtc, x)!;
    const half = rescaleIntrinsics(K, [1920, 1440], [960, 720]);
    const scaled = projectPoint(half, angledWtc, x)!;
    expect(scaled[0]).toBeCloseTo(full[0] / 2, 9);
    expect(scaled[1]).toBeCloseTo(full[1] / 2, 9);
  });

  it("reports the camera height above the plane", () => {
    expect(cameraHeightAbovePlane(angled, TABLE)).toBeCloseTo(0.5, 12);
    expect(cameraHeightAbovePlane(nadir, TABLE)).toBeCloseTo(0.45, 12);
  });

  it("converts an ARKit camera transform to the CV convention", () => {
    // Nadir ARKit camera: basis columns x=(1,0,0), y=(0,0,-1), z=(0,1,0),
    // positioned at (0, 0.45, 0). Row-major serialization:
    const arkitRowMajor = [1, 0, 0, 0, 0, 0, 1, 0.45, 0, -1, 0, 0, 0, 0, 0, 1];
    const pose = poseFromArkitCameraToWorld(arkitRowMajor);
    // CV forward (third rotation column) must point straight down.
    expect(pose.rotation[2]).toBeCloseTo(0, 12);
    expect(pose.rotation[5]).toBeCloseTo(-1, 12);
    expect(pose.rotation[8]).toBeCloseTo(0, 12);
    // A table point projects and round-trips through the ray.
    const x: Vec3 = [0.05, 0, 0.1];
    const wtc = worldToCamera(pose);
    const px = projectPoint(K, wtc, x)!;
    const ray = pixelRay(K, pose, px);
    const back = intersectRayPlane(ray.origin, ray.dir, TABLE)!;
    back.forEach((v, i) => expect(v).toBeCloseTo(x[i]!, 9));
  });
});

describe("plane frame (MATH.md §2.2, §3.1)", () => {
  it("planeCoords/planePoint round-trip", () => {
    const basis = planeBasis(TABLE);
    const x: Vec3 = [0.07, 0, -0.12];
    const uv = planeCoords(basis, x);
    const back = planePoint(basis, uv);
    back.forEach((v, i) => expect(v).toBeCloseTo(x[i]!, 12));
  });
});

describe("homography (MATH.md §3)", () => {
  const basis = planeBasis(TABLE);
  const H = planeToImageHomography(K, angledWtc, basis);
  const Hinv = mat3Inverse(H);

  it("recovers the exact metric area of a known square from pixels", () => {
    const corners: Vec3[] = [
      [-0.05, 0, -0.05],
      [0.05, 0, -0.05],
      [0.05, 0, 0.05],
      [-0.05, 0, 0.05],
    ];
    const polygonPx = corners.map((c) => projectPoint(K, angledWtc, c)!) as Vec2[];
    const area = metricPolygonAreaM2(Hinv, polygonPx);
    expect(area).toBeCloseTo(0.01, 9);
  });

  it("matches direct projection for on-plane points", () => {
    const uv = applyHomography(H, [0.03, -0.04]);
    const world = planePoint(basis, [0.03, -0.04]);
    const direct = projectPoint(K, angledWtc, world)!;
    expect(dist2d(uv, direct)).toBeLessThan(1e-9);
  });

  it("ruler residual is zero for a perfectly calibrated capture", () => {
    const p1: Vec3 = [0.1, 0, -0.02];
    const p2: Vec3 = [-0.08, 0, 0.06];
    const lengthM = Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
    const r = rulerResidualM(K, angledWtc, Hinv, { p1, p2, lengthM });
    expect(r).not.toBeNull();
    expect(r!).toBeLessThan(1e-9);
  });

  it("shows the off-plane inflation Z/(Z−h) and the correction removes it (MATH.md §3.2)", () => {
    const basisN = planeBasis(TABLE);
    const Hn = planeToImageHomography(K, nadirWtc, basisN);
    const HnInv = mat3Inverse(Hn);
    const h = 0.09;
    const r = 0.05;
    const elevated: Vec3 = [r, h, 0];
    const px = projectPoint(K, nadirWtc, elevated)!;
    const mapped = applyHomography(HnInv, px);
    const apparentRadius = Math.hypot(mapped[0], mapped[1]);
    const expectedInflation = 0.45 / (0.45 - h);
    expect(apparentRadius / r).toBeCloseTo(expectedInflation, 9);
    const corrected = apparentRadius * elevationLengthFactor(0.45, h);
    expect(corrected).toBeCloseTo(r, 9);
  });
});

describe("area and volume (MATH.md §4)", () => {
  it("shoelace area of a unit right triangle", () => {
    expect(polygonArea([[0, 0], [1, 0], [0, 1]])).toBeCloseTo(0.5, 12);
  });

  it("shape prior scales as A^(3/2)", () => {
    const a = 0.01;
    expect(volumeShapePriorM3(4 * a, 0.55)).toBeCloseTo(8 * volumeShapePriorM3(a, 0.55), 12);
  });

  it("area×height with fill factor", () => {
    expect(volumeAreaHeightM3(0.01, 0.03, 0.5)).toBeCloseTo(1.5e-4, 12);
  });

  it("height-field integration of a uniform slab equals A·h", () => {
    const heights = new Array(100).fill(0.03);
    const cell = 0.01 * 0.01;
    expect(integrateHeightFieldM3(heights, cell)).toBeCloseTo(100 * 0.03 * cell, 12);
  });

  it("mass from volume and density", () => {
    expect(massG(236.6, 0.668)).toBeCloseTo(158.05, 1);
  });
});

describe("energy (MATH.md §6)", () => {
  const rice = { kcal: 130, proteinG: 2.7, carbsG: 28, fatG: 0.3, micros: { potassiumMg: 35 } };

  it("scales per-100g values by mass", () => {
    const n = nutrientsForMassG(rice, 200);
    expect(n.kcal).toBeCloseTo(260, 9);
    expect(n.proteinG).toBeCloseTo(5.4, 9);
    expect(n.micros.potassiumMg).toBeCloseTo(70, 9);
  });

  it("accepts a consistent Atwater identity and rejects a broken one", () => {
    expect(atwaterDeviation(rice.kcal, rice.proteinG, rice.carbsG, rice.fatG)).toBeLessThan(0.15);
    expect(atwaterDeviation(500, 2.7, 28, 0.3)).toBeGreaterThan(0.15);
    expect(atwaterDeviation(0, 1, 1, 1)).toBeNull();
  });
});

describe("error budget (MATH.md §8)", () => {
  it("matches the documented v0/v1 numbers", () => {
    const priorOnly = combinedRelativeError(errorPreset("ruler", false));
    const measured = combinedRelativeError(errorPreset("ruler", true));
    expect(priorOnly).toBeCloseTo(0.306, 2);
    expect(measured).toBeCloseTo(0.207, 2);
    expect(measured).toBeLessThan(priorOnly);
  });

  it("doubles the scale term (area ∝ s²)", () => {
    const a = combinedRelativeError({ scaleRel: 0.1, segmentationRel: 0, heightRel: 0, densityRel: 0 });
    expect(a).toBeCloseTo(0.2, 12);
  });
});

describe("intrinsics parsing", () => {
  it("accepts a valid matrix and rejects a broken one", () => {
    const k = intrinsicsFromMatrix([[1500, 0, 960], [0, 1500, 720], [0, 0, 1]]);
    expect(k.fx).toBe(1500);
    expect(() => intrinsicsFromMatrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])).toThrow();
  });
});
