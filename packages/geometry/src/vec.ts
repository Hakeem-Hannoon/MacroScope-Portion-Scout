export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

/** Row-major 3×3 matrix. */
export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

/** Row-major 4×4 matrix, translation in the last column ([3], [7], [11]). */
export type Mat4 = number[];

const EPS = 1e-12;

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const dist = (a: Vec3, b: Vec3): number => norm(sub(a, b));
export const dist2d = (a: Vec2, b: Vec2): number => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function normalize(a: Vec3): Vec3 {
  const n = norm(a);
  if (n < EPS) throw new Error("cannot normalize a zero-length vector");
  return scale(a, 1 / n);
}

export const mat3Identity = (): Mat3 => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export const mat3MulVec = (m: Mat3, v: Vec3): Vec3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];

export function mat3Mul(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        a[r * 3 + 0]! * b[0 * 3 + c]! +
        a[r * 3 + 1]! * b[1 * 3 + c]! +
        a[r * 3 + 2]! * b[2 * 3 + c]!;
    }
  }
  return out as unknown as Mat3;
}

export const mat3Transpose = (m: Mat3): Mat3 => [
  m[0], m[3], m[6],
  m[1], m[4], m[7],
  m[2], m[5], m[8],
];

export const mat3FromCols = (c1: Vec3, c2: Vec3, c3: Vec3): Mat3 => [
  c1[0], c2[0], c3[0],
  c1[1], c2[1], c3[1],
  c1[2], c2[2], c3[2],
];

export function mat3Det(m: Mat3): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

export function mat3Inverse(m: Mat3): Mat3 {
  const det = mat3Det(m);
  if (Math.abs(det) < EPS) throw new Error("matrix is singular");
  const s = 1 / det;
  return [
    (m[4] * m[8] - m[5] * m[7]) * s,
    (m[2] * m[7] - m[1] * m[8]) * s,
    (m[1] * m[5] - m[2] * m[4]) * s,
    (m[5] * m[6] - m[3] * m[8]) * s,
    (m[0] * m[8] - m[2] * m[6]) * s,
    (m[2] * m[3] - m[0] * m[5]) * s,
    (m[3] * m[7] - m[4] * m[6]) * s,
    (m[1] * m[6] - m[0] * m[7]) * s,
    (m[0] * m[4] - m[1] * m[3]) * s,
  ];
}

/** Upper-left 3×3 of a row-major 4×4. */
export const mat4RotationRowMajor = (m: Mat4): Mat3 => [
  m[0]!, m[1]!, m[2]!,
  m[4]!, m[5]!, m[6]!,
  m[8]!, m[9]!, m[10]!,
];

/** Translation column of a row-major 4×4. */
export const mat4TranslationRowMajor = (m: Mat4): Vec3 => [m[3]!, m[7]!, m[11]!];
