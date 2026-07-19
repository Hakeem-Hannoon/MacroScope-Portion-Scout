/**
 * SAM "segment everything" — the pure, unit-tested core of multi-ingredient
 * segmentation (image → EVERY food region, not just the center one). SlimSAM's
 * exported decoder has a dynamic `point_batch_size` dimension, so the on-device
 * adapter encodes the frame once and prompts the decoder with a GRID of points;
 * each point proposes up to three masks. This module turns that raw grid of
 * proposals into a clean, deduplicated set of ingredient regions:
 *
 *   1. gridPointPrompts     — the P×P grid of foreground prompts (resized frame).
 *   2. maskComponentPolygon — one mask's largest blob → an exact-area polygon.
 *   3. dedupeMaskCandidates — greedy IoU non-max suppression + area filtering.
 *   4. gridPolygonToImage   — grid-space vertices → original stored-image pixels.
 *
 * Kept here (not in the RN adapter) so it is tested by vitest exactly like the
 * rest of @ppe/pipeline: the model call is injected in apps/demo/src/sam-segmenter.ts,
 * the reduction logic is proven against synthetic grids. All coordinate
 * conventions match preprocess.ts (SAM 1024 letterbox, 256×256 mask grid).
 */

import {
  SAM_INPUT_SIZE,
  type MaskBBox,
  pickBestMaskIndex,
  thresholdMaskBBox,
} from "./preprocess";

/**
 * A P×P grid of point prompts over the VALID (unpadded) image, expressed in the
 * resized SAM frame (multiply original px by `scale`), laid out as flat
 * [x0,y0, x1,y1, …] for `count` = P² points. Points sit at cell centers so the
 * grid samples the whole plate without wasting prompts on the padding. Feed a
 * chunk of these as the decoder's `input_points` ([1, chunk, 1, 2]).
 */
export function gridPointPrompts(
  origW: number,
  origH: number,
  scale: number,
  pointsPerSide: number,
): { points: Float32Array; count: number } {
  const n = Math.max(1, Math.floor(pointsPerSide));
  const points = new Float32Array(n * n * 2);
  let i = 0;
  for (let gy = 0; gy < n; gy++) {
    for (let gx = 0; gx < n; gx++) {
      // Cell-center fractions in (0,1), then into resized-frame pixels.
      const fx = (gx + 0.5) / n;
      const fy = (gy + 0.5) / n;
      points[i++] = fx * origW * scale;
      points[i++] = fy * origH * scale;
    }
  }
  return { points, count: n * n };
}

/**
 * A boolean grid marking one mask's foreground (SAM logits > threshold), plus
 * the foreground cell count. Layout matches `pred_masks` ([…masks][gh][gw]);
 * `maskIndex` selects the mask (for the batched decoder output of shape
 * [1, P, 3, gh, gw], pass `p*3 + m`).
 */
export function thresholdMaskGrid(
  logits: ArrayLike<number>,
  gridW: number,
  gridH: number,
  maskIndex: number,
  threshold = 0,
): { fg: Uint8Array; count: number } {
  const off = maskIndex * gridW * gridH;
  const fg = new Uint8Array(gridW * gridH);
  let count = 0;
  for (let p = 0; p < gridW * gridH; p++) {
    if (logits[off + p]! > threshold) {
      fg[p] = 1;
      count++;
    }
  }
  return { fg, count };
}

/**
 * Largest 4-connected foreground component of a boolean grid, returned as the
 * set of its cell indices and its bbox. Isolates the one blob a prompt landed
 * on and discards speckle — SAM masks routinely carry a few stray cells.
 */
export function largestComponent(
  fg: Uint8Array,
  gridW: number,
  gridH: number,
): { cells: Set<number>; bbox: MaskBBox } | null {
  const seen = new Uint8Array(fg.length);
  let best: { cells: Set<number>; bbox: MaskBBox } | null = null;
  const stack: number[] = [];
  for (let start = 0; start < fg.length; start++) {
    if (!fg[start] || seen[start]) continue;
    stack.length = 0;
    stack.push(start);
    seen[start] = 1;
    const cells = new Set<number>();
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    while (stack.length) {
      const idx = stack.pop()!;
      cells.add(idx);
      const x = idx % gridW;
      const y = (idx - x) / gridW;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      // 4-neighbourhood
      if (x > 0 && fg[idx - 1] && !seen[idx - 1]) (seen[idx - 1] = 1), stack.push(idx - 1);
      if (x < gridW - 1 && fg[idx + 1] && !seen[idx + 1]) (seen[idx + 1] = 1), stack.push(idx + 1);
      if (y > 0 && fg[idx - gridW] && !seen[idx - gridW])
        (seen[idx - gridW] = 1), stack.push(idx - gridW);
      if (y < gridH - 1 && fg[idx + gridW] && !seen[idx + gridW])
        (seen[idx + gridW] = 1), stack.push(idx + gridW);
    }
    if (!best || cells.size > best.cells.size) {
      best = { cells, bbox: { count: cells.size, minX, minY, maxX, maxY } };
    }
  }
  return best;
}

/**
 * Trace the outer boundary of a set of foreground cells as a rectilinear polygon
 * along grid LINES (cell corners, integer coords 0..gridW / 0..gridH). Because
 * vertices follow cell edges rather than centers, the polygon's shoelace area
 * equals the foreground cell count EXACTLY — so downstream metric area (via the
 * plane homography) is unbiased. Returns the outer ring, clockwise in grid
 * coordinates (y-down). Holes are ignored: food footprints are simply connected
 * enough that the outer ring is the right footprint.
 */
export function traceComponentPolygon(
  cells: Set<number>,
  gridW: number,
  gridH: number,
): [number, number][] {
  // Bounds MUST include x < gridW: without it, has(gridW, y) indexes
  // (y+1)*gridW — the next row's first cell — so any mask touching the right
  // column traces a wrapped, degenerate boundary (and a corrupted metric area).
  const has = (x: number, y: number): boolean =>
    x >= 0 && x < gridW && y >= 0 && y < gridH && cells.has(y * gridW + x);
  // Collect boundary unit-edges as directed corner→corner steps, oriented so the
  // foreground is on the right (clockwise outer ring in y-down space).
  const next = new Map<string, [number, number]>();
  const key = (x: number, y: number): string => `${x},${y}`;
  for (const idx of cells) {
    const x = idx % gridW;
    const y = (idx - x) / gridW;
    // top edge (x,y)->(x+1,y) when the cell above is background
    if (!has(x, y - 1)) next.set(key(x, y), [x + 1, y]);
    // right edge (x+1,y)->(x+1,y+1) when the cell to the right is background
    if (!has(x + 1, y)) next.set(key(x + 1, y), [x + 1, y + 1]);
    // bottom edge (x+1,y+1)->(x,y+1) when the cell below is background
    if (!has(x, y + 1)) next.set(key(x + 1, y + 1), [x, y + 1]);
    // left edge (x,y+1)->(x,y) when the cell to the left is background
    if (!has(x - 1, y)) next.set(key(x, y + 1), [x, y]);
  }
  if (next.size === 0) return [];
  // Start at the top-left-most boundary corner for a deterministic ring.
  let startX = Infinity;
  let startY = Infinity;
  for (const k of next.keys()) {
    const [sx, sy] = k.split(",").map(Number) as [number, number];
    if (sy < startY || (sy === startY && sx < startX)) {
      startX = sx;
      startY = sy;
    }
  }
  const ring: [number, number][] = [];
  let cx = startX;
  let cy = startY;
  for (let guard = 0; guard <= next.size; guard++) {
    ring.push([cx, cy]);
    const step = next.get(key(cx, cy));
    if (!step) break;
    [cx, cy] = step;
    if (cx === startX && cy === startY) break;
  }
  return ring;
}

/** Map a grid-corner polygon to ORIGINAL stored-image pixels (matches
 *  maskGridToImagePolygon's grid→padded-frame→÷scale→clamp mapping, per vertex). */
export function gridPolygonToImage(
  polygonGrid: [number, number][],
  gridW: number,
  gridH: number,
  scale: number,
  origW: number,
  origH: number,
  size = SAM_INPUT_SIZE,
): [number, number][] {
  return polygonGrid.map(([gx, gy]) => {
    const fx = (gx / gridW) * size;
    const fy = (gy / gridH) * size;
    return [
      Math.min(origW, Math.max(0, fx / scale)),
      Math.min(origH, Math.max(0, fy / scale)),
    ] as [number, number];
  });
}

/** One accepted mask proposal, in grid space, before mapping to image pixels. */
export interface MaskCandidate {
  polygonGrid: [number, number][];
  bbox: MaskBBox;
  /** SAM predicted-IoU score for this mask. */
  score: number;
  /** Foreground fraction of the whole grid (for area filtering). */
  coverage: number;
  /** Foreground grid-cell indices (y*gridW + x) — the true mask footprint, used
   *  for overlap-based NMS so nested/partial masks of one object collapse. */
  cells: Set<number>;
  /** Foreground cell count (= cells.size); the NMS priority (largest wins). */
  area: number;
}

/**
 * Reduce one mask's logits (already selected for a point) to a candidate: its
 * largest component's exact-area polygon, bbox, coverage, and footprint cells.
 * Null if empty.
 */
export function maskComponentPolygon(
  logits: ArrayLike<number>,
  gridW: number,
  gridH: number,
  maskIndex: number,
  score: number,
  threshold = 0,
): MaskCandidate | null {
  const { fg } = thresholdMaskGrid(logits, gridW, gridH, maskIndex, threshold);
  const comp = largestComponent(fg, gridW, gridH);
  if (!comp || comp.cells.size === 0) return null;
  const polygonGrid = traceComponentPolygon(comp.cells, gridW, gridH);
  if (polygonGrid.length < 4) return null;
  return {
    polygonGrid,
    bbox: comp.bbox,
    score,
    coverage: comp.cells.size / (gridW * gridH),
    cells: comp.cells,
    area: comp.cells.size,
  };
}

/** |A ∩ B| over two grid-cell sets (iterates the smaller for speed). */
export function cellIntersection(a: Set<number>, b: Set<number>): number {
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  let n = 0;
  for (const c of small) if (big.has(c)) n++;
  return n;
}

/** Intersection-over-union of two grid bounding boxes (inclusive cell extents). */
export function bboxIoU(a: MaskBBox, b: MaskBBox): number {
  const ix0 = Math.max(a.minX, b.minX);
  const iy0 = Math.max(a.minY, b.minY);
  const ix1 = Math.min(a.maxX, b.maxX);
  const iy1 = Math.min(a.maxY, b.maxY);
  const iw = ix1 - ix0 + 1;
  const ih = iy1 - iy0 + 1;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const areaA = (a.maxX - a.minX + 1) * (a.maxY - a.minY + 1);
  const areaB = (b.maxX - b.minX + 1) * (b.maxY - b.minY + 1);
  return inter / (areaA + areaB - inter);
}

export interface DedupeOptions {
  /**
   * Suppress a candidate that shares more than this fraction of the SMALLER
   * mask with an already-kept region (|A∩B| / min(|A|,|B|)). This is what
   * collapses the many nested/partial masks SAM's grid emits for one ingredient
   * — symmetric bbox IoU misses them because a small mask inside a big one has
   * low IoU but ~1.0 containment.
   */
  overlapThreshold?: number;
  /** Drop masks smaller than this fraction of the frame (speckle/crumbs). */
  minCoverage?: number;
  /** Drop masks larger than this fraction (the whole plate/table/scene). */
  maxCoverage?: number;
  /** Drop masks whose bbox spans ≥ this fraction of the frame in BOTH axes —
   *  SAM's whole-scene/background mask, which coverage alone can't separate from
   *  a food that fills half the frame (both ~0.5) but whose bbox is interior. */
  maxFrameSpan?: number;
  /** Drop masks whose grid bbox is thinner than this on either side (slivers). */
  minSideCells?: number;
  /** Drop masks whose bbox aspect ratio exceeds this (edge/border slivers that
   *  clear the min-side check but are still degenerate, e.g. 250×5). */
  maxAspect?: number;
  /** Hard cap on returned regions (bounds downstream classify/portion cost). */
  maxRegions?: number;
  /** Mask grid dimensions (SAM's pred_masks is 256×256); for the frame-span test. */
  gridWidth?: number;
  gridHeight?: number;
}

/**
 * Non-max suppression over the grid's mask proposals so each ingredient becomes
 * ONE region. SAM's per-point grid emits many masks per object — partial,
 * nested, and near-duplicate — so we keep the LARGEST mask of each object and
 * suppress anything that substantially overlaps it (by shared fraction of the
 * smaller mask, which catches nesting that bbox IoU misses). Whole-plate masks
 * are removed by `maxCoverage` first, so "largest wins" selects the biggest real
 * food, not the plate. Crumbs and 1-D slivers are filtered out. Returns distinct
 * regions, largest first.
 */
export function dedupeMaskCandidates(
  candidates: MaskCandidate[],
  opts: DedupeOptions = {},
): MaskCandidate[] {
  const overlapThreshold = opts.overlapThreshold ?? 0.5;
  const minCoverage = opts.minCoverage ?? 0.004;
  const maxCoverage = opts.maxCoverage ?? 0.92;
  const maxFrameSpan = opts.maxFrameSpan ?? 0.97;
  const minSideCells = opts.minSideCells ?? 4;
  const maxAspect = opts.maxAspect ?? 8;
  const maxRegions = opts.maxRegions ?? 12;
  const gridWidth = opts.gridWidth ?? 256;
  const gridHeight = opts.gridHeight ?? 256;

  const eligible = candidates
    .filter((c) => {
      if (c.coverage < minCoverage || c.coverage > maxCoverage) return false;
      const w = c.bbox.maxX - c.bbox.minX + 1;
      const h = c.bbox.maxY - c.bbox.minY + 1;
      if (w < minSideCells || h < minSideCells) return false;
      // Whole-scene mask: bbox fills the frame in both axes.
      if (w >= maxFrameSpan * gridWidth && h >= maxFrameSpan * gridHeight) return false;
      return Math.max(w, h) / Math.min(w, h) <= maxAspect;
    })
    // Largest first: represent each object by its fullest mask (best footprint
    // for the metric area), then suppress its smaller partials.
    .sort((a, b) => b.area - a.area);

  const kept: MaskCandidate[] = [];
  for (const cand of eligible) {
    if (kept.length >= maxRegions) break;
    const overlaps = kept.some((k) => {
      const inter = cellIntersection(k.cells, cand.cells);
      return inter / Math.min(k.area, cand.area) > overlapThreshold;
    });
    if (!overlaps) kept.push(cand);
  }
  return kept;
}
