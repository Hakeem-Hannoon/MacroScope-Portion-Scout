import {
  type Mat3,
  type Mat4,
  type Vec2,
  type Vec3,
  mat3MulVec,
  mat3Transpose,
  mat4RotationRowMajor,
  mat4TranslationRowMajor,
  add,
  scale,
  dot,
  normalize,
} from "./vec.js";
import type { Plane } from "./plane.js";

/**
 * Pinhole intrinsics in pixels, valid for one specific image resolution.
 * MATH.md §2.1 / §9.1.
 */
export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/** Parse a 3×3 row-of-rows intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]. */
export function intrinsicsFromMatrix(rows: number[][]): Intrinsics {
  const fx = rows[0]?.[0];
  const fy = rows[1]?.[1];
  const cx = rows[0]?.[2];
  const cy = rows[1]?.[2];
  if (
    typeof fx !== "number" || typeof fy !== "number" ||
    typeof cx !== "number" || typeof cy !== "number" ||
    fx <= 0 || fy <= 0
  ) {
    throw new Error("invalid intrinsics matrix");
  }
  return { fx, fy, cx, cy };
}

export const intrinsicsToMat3 = (k: Intrinsics): Mat3 => [
  k.fx, 0, k.cx,
  0, k.fy, k.cy,
  0, 0, 1,
];

/**
 * Intrinsics are resolution-bound: resizing the image rescales fx/cx by the
 * width ratio and fy/cy by the height ratio (MATH.md §9.1).
 */
export function rescaleIntrinsics(k: Intrinsics, from: Vec2, to: Vec2): Intrinsics {
  const sx = to[0] / from[0];
  const sy = to[1] / from[1];
  return { fx: k.fx * sx, fy: k.fy * sy, cx: k.cx * sx, cy: k.cy * sy };
}

/**
 * Camera pose in CV convention: camera x right, y down, z forward.
 * `rotation` is camera-to-world (its columns are the camera axes expressed in
 * world coordinates); `center` is the camera's position in world coordinates.
 */
export interface CameraPose {
  rotation: Mat3;
  center: Vec3;
}

/**
 * ARKit's `camera.transform` (serialized row-major) uses camera x right,
 * y UP, z BACKWARD. Negating the y and z basis columns converts it to the CV
 * convention used across this library (MATH.md §2.1 / §9.3).
 */
export function poseFromArkitCameraToWorld(rowMajor4x4: Mat4): CameraPose {
  const r = mat4RotationRowMajor(rowMajor4x4);
  const rotation: Mat3 = [
    r[0], -r[1], -r[2],
    r[3], -r[4], -r[5],
    r[6], -r[7], -r[8],
  ];
  return { rotation, center: mat4TranslationRowMajor(rowMajor4x4) };
}

/** World-to-camera extrinsics: X_cam = R·X_world + t. */
export interface WorldToCamera {
  R: Mat3;
  t: Vec3;
}

export function worldToCamera(pose: CameraPose): WorldToCamera {
  const R = mat3Transpose(pose.rotation);
  return { R, t: scale(mat3MulVec(R, pose.center), -1) };
}

/**
 * The viewing ray of a pixel: origin at the camera center, direction
 * R_cw · K⁻¹·[u,v,1]ᵀ (MATH.md §2.1).
 */
export function pixelRay(
  k: Intrinsics,
  pose: CameraPose,
  px: Vec2,
): { origin: Vec3; dir: Vec3 } {
  const dCam: Vec3 = [(px[0] - k.cx) / k.fx, (px[1] - k.cy) / k.fy, 1];
  return { origin: pose.center, dir: normalize(mat3MulVec(pose.rotation, dCam)) };
}

/** Project a world point to pixels; null when it lies behind the camera. */
export function projectPoint(k: Intrinsics, wtc: WorldToCamera, x: Vec3): Vec2 | null {
  const xc = add(mat3MulVec(wtc.R, x), wtc.t);
  if (xc[2] < 1e-9) return null;
  return [k.fx * (xc[0] / xc[2]) + k.cx, k.fy * (xc[1] / xc[2]) + k.cy];
}

/** Signed camera height above a plane (positive on the normal's side). */
export function cameraHeightAbovePlane(pose: CameraPose, plane: Plane): number {
  return dot(plane.n, pose.center) - plane.d0;
}
