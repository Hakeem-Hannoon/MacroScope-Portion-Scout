/**
 * On-device multi-ingredient segmentation via SlimSAM (a SAM 2.1-tiny-class
 * promptable model) — the roadmap P2 segmenter, in "segment everything" mode so
 * the pipeline weighs EVERY food on the plate, not just the centered one.
 *
 * The frame is encoded ONCE (the expensive vision-encoder call), then the cheap
 * mask decoder is prompted at each point of a P×P grid. Each prompt is a SINGLE
 * point with the exact [1,1,1,2] tensor shape the preprocessing was validated
 * against in Node — deliberately NOT a batched point-grid: batching the decoder
 * (point_batch_size > 1) both allocates a ~48 MB output tensor per call and
 * exercises a decoder path the exported model may not support at runtime, either
 * of which hard-crashes below the JS layer on device. One point at a time keeps
 * each output ~0.8 MB and uses only the proven call. Each prompt proposes up to
 * three masks; the best is reduced to its largest connected component's
 * exact-area polygon, and greedy IoU non-max suppression collapses the many grid
 * points that hit one ingredient into a single region.
 *
 * All of that reduction is the pure, unit-tested code in @ppe/pipeline
 * (gridPointPrompts / maskComponentPolygon / dedupeMaskCandidates /
 * gridPolygonToImage); this file owns only the RN-specific model I/O. Downstream
 * is unchanged: estimateMeal already classifies + portions every region, so the
 * result carries one item per ingredient. On any failure it falls back to a
 * centered square so classification + metric geometry still run (roadmap P1).
 * See docs/REAL_ADAPTERS.md.
 */
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import {
  type MaskCandidate,
  SAM_INPUT_SIZE,
  type Region,
  type Segmenter,
  dedupeMaskCandidates,
  gridPointPrompts,
  gridPolygonToImage,
  maskComponentPolygon,
  packSamTensor,
  pickBestMaskIndex,
  samResizeTarget,
  thresholdMaskBBox,
} from "@ppe/pipeline";
import { decodeJpegRgba, manipulateToBase64 } from "./image-io";

/** Tunables for the automatic mask generation sweep (perf ⇄ recall). */
export interface SamEverythingOptions {
  /** Grid density: pointsPerSide² single-point prompts (= decoder calls). */
  pointsPerSide?: number;
  /** Suppress a mask sharing more than this fraction of the smaller mask with a
   *  kept one — collapses the many partial masks SAM emits per ingredient. */
  overlapThreshold?: number;
  /** Drop masks below this frame fraction (crumbs/fragments) / above it (plate). */
  minCoverage?: number;
  maxCoverage?: number;
  /** Hard cap on ingredients returned (bounds classify/portion cost). */
  maxRegions?: number;
}

const DEFAULTS: Required<SamEverythingOptions> = {
  // 8×8 = 64 decoder calls: covers a mixed plate while staying responsive on a
  // first device bring-up. Raise for more recall on small/garnish items.
  pointsPerSide: 8,
  overlapThreshold: 0.5,
  // 1.2% of the frame: below this a "region" is a fragment/crumb, not a food
  // worth logging — filters the noise masks SAM's grid throws off on busy plates.
  minCoverage: 0.012,
  maxCoverage: 0.92,
  maxRegions: 12,
};

/** The placeholder region — a centered square (~40% of the frame) — used as a
 *  graceful fallback. Exercises the real metric geometry against a weighable
 *  single dish (roadmap P1) even when the model can't run. */
function centeredSquare(w: number, h: number): Region[] {
  const side = Math.min(w, h) * 0.4;
  const cx = w / 2;
  const cy = h / 2;
  return [
    {
      polygonPx: [
        [cx - side / 2, cy - side / 2],
        [cx + side / 2, cy - side / 2],
        [cx + side / 2, cy + side / 2],
        [cx - side / 2, cy + side / 2],
      ],
    },
  ];
}

export function createSamSegmenter(
  vision: InferenceSession,
  decoder: InferenceSession,
  options: SamEverythingOptions = {},
): Segmenter {
  const opts = { ...DEFAULTS, ...options };
  return {
    async segment(imageUri, [W, H]) {
      try {
        const { scale, newW, newH } = samResizeTarget(W, H);
        const base64 = await manipulateToBase64(imageUri, { width: newW, height: newH });
        const rgba = decodeJpegRgba(base64);
        const pixels = packSamTensor(rgba, SAM_INPUT_SIZE);

        const encoded = await vision.run({
          pixel_values: new Tensor("float32", pixels, [1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE]),
        });
        console.log("[SAM] encoded; sweeping grid");

        const { points, count } = gridPointPrompts(W, H, scale, opts.pointsPerSide);
        const candidates: MaskCandidate[] = [];
        let gridW = 256;
        let gridH = 256;

        for (let i = 0; i < count; i++) {
          let decoded: Awaited<ReturnType<typeof decoder.run>>;
          try {
            decoded = await decoder.run({
              // Single point, the shape validated in Node: [1,1,1,2].
              input_points: new Tensor("float32", [points[i * 2]!, points[i * 2 + 1]!], [1, 1, 1, 2]),
              // int64 foreground label; Hermes (RN 0.79) supports BigInt64Array.
              input_labels: new Tensor("int64", new BigInt64Array([1n]), [1, 1, 1]),
              image_embeddings: encoded.image_embeddings,
              image_positional_embeddings: encoded.image_positional_embeddings,
            });
          } catch (err) {
            // A failure at the very first prompt is systematic (bad shapes /
            // unsupported op) — bail to the fallback rather than repeat it 63×.
            if (i === 0) throw err;
            console.warn(`[SAM] decoder failed at point ${i}; skipping`, err);
            continue;
          }

          const pred = decoded.pred_masks; // dims [1, 1, numMasks, gh, gw]
          const iou = decoded.iou_scores; // dims [1, 1, numMasks]
          const numMasks = pred.dims[2]!;
          gridH = pred.dims[3]!;
          gridW = pred.dims[4]!;
          const logits = pred.data as Float32Array;
          const iouData = iou.data as Float32Array;
          const cells = gridW * gridH;

          // Pick the best of this point's masks (highest IoU under the
          // whole-plate coverage cap), then reduce it to a polygon.
          const iouTriple: number[] = [];
          const coverage: number[] = [];
          for (let m = 0; m < numMasks; m++) {
            iouTriple.push(iouData[m]!);
            coverage.push(thresholdMaskBBox(logits, gridW, gridH, m).count / cells);
          }
          const best = pickBestMaskIndex(iouTriple, coverage);
          const cand = maskComponentPolygon(logits, gridW, gridH, best, iouTriple[best]!);
          if (cand) candidates.push(cand);
        }

        const kept = dedupeMaskCandidates(candidates, { ...opts, gridWidth: gridW, gridHeight: gridH });
        console.log(`[SAM] ${candidates.length} candidates → ${kept.length} regions`);
        if (kept.length === 0) return centeredSquare(W, H);

        return kept.map((c) => ({
          polygonPx: gridPolygonToImage(c.polygonGrid, gridW, gridH, scale, W, H),
        }));
      } catch (error) {
        console.warn("[SAM] segmentation failed; centered-square fallback:", error);
        return centeredSquare(W, H);
      }
    },
  };
}
