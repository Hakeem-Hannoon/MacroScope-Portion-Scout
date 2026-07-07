import {
  type Vec2,
  type Vec3,
  add,
  sub,
  scale,
  dot,
  cross,
  normalize,
} from "./vec.js";

/** The plane { X : n·X = d0 } with unit normal n. MATH.md §2.2. */
export interface Plane {
  n: Vec3;
  d0: number;
}

/**
 * Ray–plane intersection (MATH.md §2.2):
 *   t* = (d0 − n·o) / (n·d),  P = o + t*·d
 * Returns null for rays parallel to the plane or intersections behind the
 * ray origin.
 */
export function intersectRayPlane(origin: Vec3, dir: Vec3, plane: Plane): Vec3 | null {
  const denom = dot(plane.n, dir);
  if (Math.abs(denom) < 1e-9) return null;
  const t = (plane.d0 - dot(plane.n, origin)) / denom;
  if (t <= 0) return null;
  return add(origin, scale(dir, t));
}

/** An orthonormal 2D frame on a plane: origin O, in-plane axes e1 ⊥ e2 ⊥ n. */
export interface PlaneBasis {
  O: Vec3;
  e1: Vec3;
  e2: Vec3;
  n: Vec3;
}

/**
 * Build a plane frame (MATH.md §3.1). The default origin is the plane point
 * closest to the world origin, n·d0.
 */
export function planeBasis(plane: Plane, origin?: Vec3): PlaneBasis {
  const n = normalize(plane.n);
  const O = origin ?? scale(n, plane.d0);
  const seed: Vec3 = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = normalize(sub(seed, scale(n, dot(seed, n))));
  const e2 = cross(n, e1);
  return { O, e1, e2, n };
}

/** World point → 2D plane coordinates (meters). */
export function planeCoords(basis: PlaneBasis, x: Vec3): Vec2 {
  const d = sub(x, basis.O);
  return [dot(d, basis.e1), dot(d, basis.e2)];
}

/** 2D plane coordinates (meters) → world point. */
export function planePoint(basis: PlaneBasis, p: Vec2): Vec3 {
  return add(basis.O, add(scale(basis.e1, p[0]), scale(basis.e2, p[1])));
}
