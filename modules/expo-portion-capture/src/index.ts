import { requireNativeModule } from "expo-modules-core";

/**
 * Types mirror `capturePayloadSchema` in @ppe/pipeline (the source of truth).
 * Validate every payload with that schema before running the pipeline —
 * the native side is treated as an untrusted producer.
 */
export interface CaptureStroke {
  p1: [number, number, number];
  p2: [number, number, number];
  length_m: number;
  kind: "horizontal" | "vertical";
}

export interface CapturePayload {
  version: 1;
  image: string;
  image_size: [number, number];
  intrinsics: number[][];
  camera_to_world: number[];
  plane: { normal: [number, number, number]; d0: number; extent?: [number, number] };
  strokes: CaptureStroke[];
  depth: {
    map: string;
    confidence: string | null;
    size: [number, number];
    intrinsics: number[][];
  } | null;
  tracking?: { state: string; plane_source: string };
  scale_source: "lidar" | "ruler" | "reference_object" | "stated" | "none";
}

export interface LaunchOptions {
  /** Require at least one horizontal stroke before the shutter enables. Default true. */
  requireStroke?: boolean;
  /** Minimum accepted stroke length in meters. Default 0.10 (MATH.md §2.3). */
  minStrokeLengthM?: number;
}

const native = requireNativeModule("ExpoPortionCapture");

/** True when the device supports ARKit world tracking. */
export function isSupported(): boolean {
  return native.isSupported();
}

/**
 * Presents the full-screen AR capture UI. Resolves with a CapturePayload,
 * or null when the user cancels. Requires a development build (ARKit is
 * unavailable in Expo Go) and the NSCameraUsageDescription Info.plist key.
 */
export async function launch(options: LaunchOptions = {}): Promise<CapturePayload | null> {
  const payload = await native.launch({
    requireStroke: options.requireStroke ?? true,
    minStrokeLengthM: options.minStrokeLengthM ?? 0.1,
  });
  return (payload as CapturePayload | null) ?? null;
}
