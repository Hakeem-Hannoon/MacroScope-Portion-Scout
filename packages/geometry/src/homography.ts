import {
  type Mat3,
  type Vec2,
  type Vec3,
  add,
  mat3FromCols,
  mat3Mul,
  mat3MulVec,
  dist2d,
} from "./vec.js";
import {
  type Intrinsics,
  type WorldToCamera,
  intrinsicsToMat3,
  projectPoint,
} from "./camera.js";
import type { PlaneBasis } from "./plane.js";

/**
 * Homography from metric plane coordinates (x, y) to image pixels
 * (MATH.md §3.1):
 *
 *   p ~ K · [ R·e1 | R·e2 | R·O + t ] · [x, y, 1]ᵀ
 *
 * Invert it (mat3Inverse) to map image pixels of on-plane points to meters.
 */
export function planeToImageHomography(
  k: Intrinsics,
  wtc: WorldToCamera,
  basis: PlaneBasis,
): Mat3 {
  const c1 = mat3MulVec(wtc.R, basis.e1);
  const c2 = mat3MulVec(wtc.R, basis.e2);
  const c3 = add(mat3MulVec(wtc.R, basis.O), wtc.t);
  return mat3Mul(intrinsicsToMat3(k), mat3FromCols(c1, c2, c3));
}

/** Apply a homography to a 2D point (projective dehomogenization). */
export function applyHomography(h: Mat3, p: Vec2): Vec2 {
  const v = mat3MulVec(h, [p[0], p[1], 1]);
  if (Math.abs(v[2]) < 1e-12) throw new Error("point maps to infinity");
  return [v[0] / v[2], v[1] / v[2]];
}

/**
 * Self-check of the whole calibration chain (MATH.md §3.1): project the
 * ruler's 3D endpoints into the image, map them back through the inverse
 * homography, and compare the recovered length with the measured one.
 * Returns the absolute residual in meters, or null when an endpoint is
 * unprojectable.
 */
export function rulerResidualM(
  k: Intrinsics,
  wtc: WorldToCamera,
  imageToPlane: Mat3,
  stroke: { p1: Vec3; p2: Vec3; lengthM: number },
): number | null {
  const q1 = projectPoint(k, wtc, stroke.p1);
  const q2 = projectPoint(k, wtc, stroke.p2);
  if (!q1 || !q2) return null;
  const a = applyHomography(imageToPlane, q1);
  const b = applyHomography(imageToPlane, q2);
  return Math.abs(dist2d(a, b) - stroke.lengthM);
}

/**
 * Off-plane (elevation) length correction, MATH.md §3.2: a feature at height
 * h above the reference plane, seen by a camera at height Z, appears scaled
 * by Z/(Z−h) when mapped through the plane's homography. Multiply apparent
 * lengths by this factor to correct; square it for areas.
 */
export function elevationLengthFactor(cameraHeightM: number, featureHeightM: number): number {
  if (cameraHeightM <= 0) throw new Error("camera must be above the plane");
  if (featureHeightM <= 0) return 1;
  if (featureHeightM >= cameraHeightM) throw new Error("feature height exceeds camera height");
  return (cameraHeightM - featureHeightM) / cameraHeightM;
}
