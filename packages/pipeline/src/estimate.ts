import {
  type Micros,
  type Plane,
  type Vec3,
  applyHomography,
  atwaterDeviation,
  cameraHeightAbovePlane,
  combinedRelativeError,
  elevationLengthFactor,
  errorPreset,
  intrinsicsFromMatrix,
  integrateHeightFieldM3,
  M2_TO_CM2,
  M3_TO_ML,
  massG,
  mat3Inverse,
  metricPolygonAreaM2,
  normalize,
  nutrientsForMassG,
  planeBasis,
  planeToImageHomography,
  poseFromArkitCameraToWorld,
  rulerResidualM,
  volumeAreaHeightM3,
  volumeShapePriorM3,
  worldToCamera,
} from "@ppe/geometry";
import {
  type CapturePayload,
  type EstimateItem,
  type EstimateResult,
  capturePayloadSchema,
  estimateResultSchema,
} from "./contracts.js";
import type {
  Classifier,
  DepthProvider,
  FoodRecord,
  NutrientStore,
  Segmenter,
} from "./adapters.js";

export interface EstimateDeps {
  segmenter: Segmenter;
  classifier: Classifier;
  nutrients: NutrientStore;
  depth?: DepthProvider;
}

/** Placeholder mound constant until the Nutrition5k prior fit lands (model/). */
export const DEFAULT_KAPPA = 0.55;
const DEFAULT_MOUND_PHI = 0.58;
const LOW_CONFIDENCE = 0.5;
const ATWATER_TOLERANCE = 0.15;

const round = (x: number, places: number): number => {
  const f = 10 ** places;
  return Math.round(x * f) / f;
};

/**
 * The end-to-end estimate: capture payload in, calories/macros/micros out.
 * Model inference happens behind the adapters; everything metric happens
 * here, in plain code, exactly as derived in docs/MATH.md.
 */
export async function estimateMeal(
  payloadInput: unknown,
  deps: EstimateDeps,
): Promise<EstimateResult> {
  const payload: CapturePayload = capturePayloadSchema.parse(payloadInput);

  const k = intrinsicsFromMatrix(payload.intrinsics);
  const pose = poseFromArkitCameraToWorld(payload.camera_to_world);
  const wtc = worldToCamera(pose);
  const plane: Plane = {
    n: normalize(payload.plane.normal as Vec3),
    d0: payload.plane.d0,
  };
  const basis = planeBasis(plane);
  const imageToPlane = mat3Inverse(planeToImageHomography(k, wtc, basis));

  const rawCamHeight = cameraHeightAbovePlane(pose, plane);
  const camHeightM = rawCamHeight > 0.05 ? rawCamHeight : null;

  // Calibration self-check over the horizontal strokes (MATH.md §3.1).
  let residualM: number | null = null;
  for (const stroke of payload.strokes) {
    if (stroke.kind !== "horizontal") continue;
    const r = rulerResidualM(k, wtc, imageToPlane, {
      p1: stroke.p1 as Vec3,
      p2: stroke.p2 as Vec3,
      lengthM: stroke.length_m,
    });
    if (r !== null) residualM = Math.max(residualM ?? 0, r);
  }

  const measuredHeightM =
    payload.strokes.find((s) => s.kind === "vertical")?.length_m ?? null;

  const regions = await deps.segmenter.segment(payload.image, payload.image_size);

  const items: EstimateItem[] = [];
  let heightWasMeasured = measuredHeightM !== null;

  for (const region of regions) {
    const cls = await deps.classifier.classify(payload.image, region);
    const record = await deps.nutrients.lookup(cls.label);
    const flags: string[] = [];
    if (!record) flags.push("no_db_match");
    if (cls.confidence < LOW_CONFIDENCE) flags.push("low_confidence");
    if (payload.scale_source === "none") flags.push("no_scale");

    const shape = record?.shape ?? { kind: "mound" as const, kappa: DEFAULT_KAPPA };
    const kappa = shape.kappa ?? DEFAULT_KAPPA;

    let areaM2 = metricPolygonAreaM2(imageToPlane, region.polygonPx);
    let volumeM3: number;
    let heightM: number | null = null;
    let method: EstimateItem["geometry"]["method"];

    if (deps.depth && payload.depth) {
      // Route (a), MATH.md §4a: integrate the measured height field.
      const hf = await deps.depth.heightField(payload, region);
      volumeM3 = integrateHeightFieldM3(hf.heights, hf.cellAreaM2);
      heightM = hf.maxHeightM ?? null;
      method = "lidar_integration";
      heightWasMeasured = true;
    } else {
      heightM =
        measuredHeightM ?? (shape.kind === "flat" ? shape.hBarM ?? null : null);

      // Off-plane bias correction at the food's mid-height (MATH.md §3.2).
      const hForCorrection = heightM ?? kappa * Math.sqrt(areaM2);
      if (camHeightM !== null && hForCorrection > 0 && hForCorrection < camHeightM) {
        const f = elevationLengthFactor(camHeightM, hForCorrection / 2);
        areaM2 *= f * f;
      }

      if (heightM !== null) {
        const phi = shape.phi ?? (shape.kind === "flat" ? 1 : DEFAULT_MOUND_PHI);
        volumeM3 = volumeAreaHeightM3(areaM2, heightM, phi);
        method = "area_height";
      } else {
        volumeM3 = volumeShapePriorM3(areaM2, kappa);
        method = "shape_prior";
      }
      if (shape.kind === "container") {
        method = "container_prior";
        flags.push("container_prior_used");
      }
    }

    const volumeMl = volumeM3 * M3_TO_ML;
    const item = record
      ? buildMatchedItem(cls.label, cls.confidence, record, volumeMl, flags)
      : {
          label: cls.label,
          confidence: cls.confidence,
          mass_g: null,
          kcal: null,
          protein_g: null,
          carbs_g: null,
          fat_g: null,
          micros: null,
          flags,
        };

    items.push({
      ...item,
      geometry: {
        area_cm2: round(areaM2 * M2_TO_CM2, 1),
        height_cm: heightM !== null ? round(heightM * 100, 1) : null,
        volume_ml: round(volumeMl, 1),
        method,
      },
    });
  }

  const result: EstimateResult = {
    items,
    totals: sumTotals(items),
    quality: {
      scale_source: payload.scale_source,
      ruler_residual_mm: residualM !== null ? round(residualM * 1000, 2) : null,
      est_relative_error: round(
        combinedRelativeError(errorPreset(payload.scale_source, heightWasMeasured)),
        3,
      ),
      camera_height_m: camHeightM !== null ? round(camHeightM, 3) : null,
    },
  };

  // The output contract is enforced, exactly like the input contract.
  return estimateResultSchema.parse(result);
}

function buildMatchedItem(
  label: string,
  confidence: number,
  record: FoodRecord,
  volumeMl: number,
  flags: string[],
): Omit<EstimateItem, "geometry"> {
  const mass = massG(volumeMl, record.densityGPerMl);
  const n = nutrientsForMassG(record.per100, mass);
  const deviation = atwaterDeviation(
    record.per100.kcal,
    record.per100.proteinG,
    record.per100.carbsG,
    record.per100.fatG,
  );
  if (deviation !== null && deviation > ATWATER_TOLERANCE) {
    flags.push("atwater_mismatch");
  }
  const micros: Micros = {};
  for (const [key, value] of Object.entries(n.micros)) {
    micros[key as keyof Micros] = round(value, key.endsWith("Mg") ? 0 : 1);
  }
  return {
    label,
    confidence,
    mass_g: round(mass, 1),
    kcal: round(n.kcal, 0),
    protein_g: round(n.proteinG, 1),
    carbs_g: round(n.carbsG, 1),
    fat_g: round(n.fatG, 1),
    micros,
    flags,
  };
}

function sumTotals(items: EstimateItem[]): EstimateResult["totals"] {
  const micros: Micros = {};
  let kcal = 0;
  let protein = 0;
  let carbs = 0;
  let fat = 0;
  for (const item of items) {
    kcal += item.kcal ?? 0;
    protein += item.protein_g ?? 0;
    carbs += item.carbs_g ?? 0;
    fat += item.fat_g ?? 0;
    for (const [key, value] of Object.entries(item.micros ?? {})) {
      if (typeof value !== "number") continue;
      const mk = key as keyof Micros;
      micros[mk] = (micros[mk] ?? 0) + value;
    }
  }
  for (const [key, value] of Object.entries(micros)) {
    micros[key as keyof Micros] = Math.round((value as number) * 10) / 10;
  }
  return {
    kcal: Math.round(kcal),
    protein_g: Math.round(protein * 10) / 10,
    carbs_g: Math.round(carbs * 10) / 10,
    fat_g: Math.round(fat * 10) / 10,
    micros,
  };
}
