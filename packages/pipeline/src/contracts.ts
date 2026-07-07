import { z } from "zod";

/**
 * The versioned contract between the native capture module and the pipeline.
 * Everything the geometry needs travels in this document; binary assets
 * (image, depth) travel as file URIs next to it. See docs/ARCHITECTURE.md.
 */

const vec3 = z.tuple([z.number(), z.number(), z.number()]);
const size2 = z.tuple([z.number().int().positive(), z.number().int().positive()]);
const mat3Rows = z.array(z.array(z.number()).length(3)).length(3);

export const strokeSchema = z.object({
  /** World-space endpoints in meters (ARKit world frame). */
  p1: vec3,
  p2: vec3,
  length_m: z.number().positive(),
  /** horizontal = on the table plane (scale); vertical = up the food (height). */
  kind: z.enum(["horizontal", "vertical"]),
});

export const scaleSourceSchema = z.enum([
  "lidar",
  "ruler",
  "reference_object",
  "stated",
  "none",
]);

export const capturePayloadSchema = z.object({
  version: z.literal(1),
  /** URI/path of the captured image (sensor orientation, full resolution). */
  image: z.string().min(1),
  image_size: size2,
  /** Intrinsics valid for image_size — rescale on any resize (MATH.md §9.1). */
  intrinsics: mat3Rows,
  /** Camera-to-world, row-major 4×4, ARKit camera convention (y up, z backward). */
  camera_to_world: z.array(z.number()).length(16),
  /** The supporting surface: n·X = d0 in the world frame. */
  plane: z.object({
    normal: vec3,
    d0: z.number(),
    extent: z.tuple([z.number(), z.number()]).optional(),
  }),
  strokes: z.array(strokeSchema).max(8),
  /** LiDAR/scene depth at capture time; null on devices without it. */
  depth: z
    .object({
      map: z.string(),
      confidence: z.string().nullable(),
      size: size2,
      intrinsics: mat3Rows,
    })
    .nullable(),
  tracking: z
    .object({ state: z.string(), plane_source: z.string() })
    .optional(),
  scale_source: scaleSourceSchema,
});

export type CapturePayload = z.infer<typeof capturePayloadSchema>;
export type Stroke = z.infer<typeof strokeSchema>;

export const microsSchema = z
  .object({
    fiberG: z.number().nonnegative(),
    sugarG: z.number().nonnegative(),
    satFatG: z.number().nonnegative(),
    sodiumMg: z.number().nonnegative(),
    cholesterolMg: z.number().nonnegative(),
    potassiumMg: z.number().nonnegative(),
    calciumMg: z.number().nonnegative(),
    ironMg: z.number().nonnegative(),
  })
  .partial();

export const estimateItemSchema = z.object({
  label: z.string(),
  confidence: z.number().min(0).max(1),
  geometry: z.object({
    area_cm2: z.number().nonnegative(),
    height_cm: z.number().nonnegative().nullable(),
    volume_ml: z.number().nonnegative(),
    method: z.enum([
      "lidar_integration",
      "area_height",
      "shape_prior",
      "container_prior",
    ]),
  }),
  /** Null when the label found no database match — the UI must ask, never guess. */
  mass_g: z.number().nonnegative().nullable(),
  kcal: z.number().nonnegative().nullable(),
  protein_g: z.number().nonnegative().nullable(),
  carbs_g: z.number().nonnegative().nullable(),
  fat_g: z.number().nonnegative().nullable(),
  micros: microsSchema.nullable(),
  flags: z.array(z.string()),
});

export const estimateResultSchema = z.object({
  items: z.array(estimateItemSchema),
  totals: z.object({
    kcal: z.number().nonnegative(),
    protein_g: z.number().nonnegative(),
    carbs_g: z.number().nonnegative(),
    fat_g: z.number().nonnegative(),
    micros: microsSchema,
  }),
  quality: z.object({
    scale_source: scaleSourceSchema,
    /** Calibration self-check (MATH.md §3.1); null without a horizontal stroke. */
    ruler_residual_mm: z.number().nonnegative().nullable(),
    /** Propagated relative error of per-item mass (MATH.md §8). */
    est_relative_error: z.number().positive(),
    camera_height_m: z.number().positive().nullable(),
  }),
});

export type EstimateItem = z.infer<typeof estimateItemSchema>;
export type EstimateResult = z.infer<typeof estimateResultSchema>;
