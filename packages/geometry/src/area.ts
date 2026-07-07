import { type Mat3, type Vec2 } from "./vec.js";
import { applyHomography } from "./homography.js";

/** Shoelace polygon area (MATH.md §3.1). Vertices in order, any winding. */
export function polygonArea(pts: Vec2[]): number {
  if (pts.length < 3) return 0;
  let acc = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]!;
    const b = pts[(i + 1) % pts.length]!;
    acc += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(acc) / 2;
}

/** Map a pixel polygon through an image→plane homography to metric meters. */
export function pixelPolygonToPlane(imageToPlane: Mat3, polygonPx: Vec2[]): Vec2[] {
  return polygonPx.map((p) => applyHomography(imageToPlane, p));
}

/** Metric area (m²) of a pixel polygon lying on the calibrated plane. */
export function metricPolygonAreaM2(imageToPlane: Mat3, polygonPx: Vec2[]): number {
  return polygonArea(pixelPolygonToPlane(imageToPlane, polygonPx));
}

export const M2_TO_CM2 = 1e4;
