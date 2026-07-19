import { describe, expect, it } from "vitest";
import {
  type MaskCandidate,
  bboxIoU,
  dedupeMaskCandidates,
  gridPointPrompts,
  gridPolygonToImage,
  largestComponent,
  maskComponentPolygon,
  thresholdMaskGrid,
  traceComponentPolygon,
} from "../src/segment-all";

/** Absolute shoelace area of a closed ring (grid units). */
function shoelaceArea(ring: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x0, y0] = ring[i]!;
    const [x1, y1] = ring[(i + 1) % ring.length]!;
    a += x0 * y1 - x1 * y0;
  }
  return Math.abs(a) / 2;
}

/** Build a boolean grid with the given foreground cells set. */
function grid(w: number, h: number, cells: [number, number][]): Uint8Array {
  const fg = new Uint8Array(w * h);
  for (const [x, y] of cells) fg[y * w + x] = 1;
  return fg;
}

describe("gridPointPrompts", () => {
  it("places P² prompts at cell centers in the resized frame", () => {
    const { points, count } = gridPointPrompts(100, 100, 1, 2);
    expect(count).toBe(4);
    // cell centers at 0.25 and 0.75 of 100
    expect(Array.from(points)).toEqual([25, 25, 75, 25, 25, 75, 75, 75]);
  });

  it("applies the SAM resize scale to prompt coordinates", () => {
    const { points } = gridPointPrompts(200, 100, 0.5, 1);
    // single center prompt at (100,50) original → ×0.5 resized
    expect(Array.from(points)).toEqual([50, 25]);
  });
});

describe("thresholdMaskGrid", () => {
  it("thresholds at 0 and counts foreground, honoring the mask offset", () => {
    // two masks of a 2×2 grid; mask 1 has three positive cells
    const gw = 2;
    const gh = 2;
    const logits = new Float32Array([
      -1, -1, -1, -1, // mask 0: all background
      5, 5, -1, 5, // mask 1: three foreground
    ]);
    const { fg, count } = thresholdMaskGrid(logits, gw, gh, 1);
    expect(count).toBe(3);
    expect(Array.from(fg)).toEqual([1, 1, 0, 1]);
  });
});

describe("largestComponent", () => {
  it("selects the largest 4-connected blob and discards speckle", () => {
    // a 3-cell blob and a 1-cell speck in an 8×8 grid
    const fg = grid(8, 8, [
      [1, 1],
      [2, 1],
      [1, 2], // blob (3 cells)
      [6, 6], // speck
    ]);
    const comp = largestComponent(fg, 8, 8)!;
    expect(comp.cells.size).toBe(3);
    expect(comp.bbox).toMatchObject({ minX: 1, minY: 1, maxX: 2, maxY: 2 });
  });

  it("does not bridge diagonally (4-connectivity)", () => {
    const fg = grid(4, 4, [
      [0, 0],
      [1, 1], // diagonal-only touch → separate components
    ]);
    const comp = largestComponent(fg, 4, 4)!;
    expect(comp.cells.size).toBe(1);
  });

  it("returns null for an empty grid", () => {
    expect(largestComponent(new Uint8Array(16), 4, 4)).toBeNull();
  });
});

describe("traceComponentPolygon", () => {
  it("traces a rectangle to a ring whose area equals the cell count", () => {
    // cells x∈{2,3,4}, y∈{2,3} → 6 cells
    const cells = new Set<number>();
    for (let y = 2; y <= 3; y++) for (let x = 2; x <= 4; x++) cells.add(y * 8 + x);
    const ring = traceComponentPolygon(cells, 8, 8);
    expect(shoelaceArea(ring)).toBe(6);
  });

  it("handles a concave (L-shaped) blob with exact area", () => {
    // 3×3 block minus the top-right cell (2,0) → 8 cells, non-convex
    const cells = new Set<number>();
    for (let y = 0; y <= 2; y++)
      for (let x = 0; x <= 2; x++) if (!(x === 2 && y === 0)) cells.add(y * 8 + x);
    const ring = traceComponentPolygon(cells, 8, 8);
    expect(cells.size).toBe(8);
    expect(shoelaceArea(ring)).toBe(8);
  });

  it("traces a mask touching the right/bottom edges without wrapping (regression)", () => {
    // A block flush against the last column (x=gridW-1) previously made has()
    // wrap to the next row, collapsing the polygon to a degenerate sliver.
    const gw = 6;
    const gh = 6;
    const cells = new Set<number>();
    for (let y = 2; y <= 5; y++) for (let x = 3; x <= 5; x++) cells.add(y * gw + x); // 3×4 = 12
    const ring = traceComponentPolygon(cells, gw, gh);
    expect(shoelaceArea(ring)).toBe(12);
    // Full-grid mask: bbox must span the whole grid, not collapse.
    const full = new Set<number>();
    for (let i = 0; i < gw * gh; i++) full.add(i);
    expect(shoelaceArea(traceComponentPolygon(full, gw, gh))).toBe(gw * gh);
  });
});

describe("gridPolygonToImage", () => {
  it("maps grid corners through the letterbox scale to original pixels", () => {
    const poly: [number, number][] = [
      [0, 0],
      [256, 0],
      [256, 256],
    ];
    const mapped = gridPolygonToImage(poly, 256, 256, 0.5, 2048, 2048, 1024);
    // (256/256)*1024 = 1024 padded → /0.5 = 2048 original, clamped to width
    expect(mapped).toEqual([
      [0, 0],
      [2048, 0],
      [2048, 2048],
    ]);
  });
});

describe("bboxIoU", () => {
  it("computes intersection-over-union of inclusive boxes", () => {
    const a = { count: 16, minX: 0, minY: 0, maxX: 3, maxY: 3 };
    const b = { count: 16, minX: 2, minY: 2, maxX: 5, maxY: 5 };
    // inter 2×2=4, union 16+16−4=28
    expect(bboxIoU(a, b)).toBeCloseTo(4 / 28, 6);
  });

  it("is zero for disjoint boxes", () => {
    const a = { count: 1, minX: 0, minY: 0, maxX: 0, maxY: 0 };
    const b = { count: 1, minX: 5, minY: 5, maxX: 5, maxY: 5 };
    expect(bboxIoU(a, b)).toBe(0);
  });
});

describe("maskComponentPolygon", () => {
  it("reduces a mask's logits to an exact-area candidate", () => {
    // 8×8 grid, mask index 1, a 3×3 positive blob at x,y ∈ {3,4,5}
    const gw = 8;
    const gh = 8;
    const logits = new Float32Array(2 * gw * gh).fill(-1);
    const off = 1 * gw * gh;
    for (let y = 3; y <= 5; y++) for (let x = 3; x <= 5; x++) logits[off + y * gw + x] = 9;
    const cand = maskComponentPolygon(logits, gw, gh, 1, 0.9)!;
    expect(cand.score).toBe(0.9);
    expect(cand.coverage).toBeCloseTo(9 / 64, 6);
    expect(cand.area).toBe(9);
    expect(cand.cells.size).toBe(9);
    expect(shoelaceArea(cand.polygonGrid)).toBe(9);
    expect(cand.bbox).toMatchObject({ minX: 3, minY: 3, maxX: 5, maxY: 5 });
  });

  it("returns null when the mask is empty", () => {
    const logits = new Float32Array(64).fill(-1);
    expect(maskComponentPolygon(logits, 8, 8, 0, 0.5)).toBeNull();
  });
});

describe("dedupeMaskCandidates", () => {
  const GRID = 100;
  const GRID_CELLS = GRID * GRID;
  // A solid rectangle x∈[x0,x1], y∈[y0,y1] as a real candidate (cells + area).
  const rect = (x0: number, y0: number, x1: number, y1: number, score = 0.9): MaskCandidate => {
    const cells = new Set<number>();
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) cells.add(y * GRID + x);
    return {
      polygonGrid: [
        [x0, y0],
        [x1 + 1, y0],
        [x1 + 1, y1 + 1],
        [x0, y1 + 1],
      ],
      bbox: { count: cells.size, minX: x0, minY: y0, maxX: x1, maxY: y1 },
      score,
      coverage: cells.size / GRID_CELLS,
      cells,
      area: cells.size,
    };
  };

  it("collapses a partial mask nested inside a larger one (keeps the larger)", () => {
    const big = rect(10, 10, 49, 49); // 40×40 = 1600 cells
    const small = rect(15, 15, 24, 24, 0.99); // 10×10 inside big, higher score
    const other = rect(60, 60, 79, 79); // disjoint object
    const kept = dedupeMaskCandidates([small, big, other]);
    expect(kept).toHaveLength(2);
    // The larger mask represents the object, despite the partial's higher score.
    expect(kept[0]!.area).toBe(1600);
    expect(kept.map((k) => k.bbox.minX).sort((a, b) => a - b)).toEqual([10, 60]);
  });

  it("keeps two adjacent low-overlap masks as distinct ingredients", () => {
    const a = rect(10, 10, 29, 29);
    const b = rect(60, 10, 79, 29);
    expect(dedupeMaskCandidates([a, b])).toHaveLength(2);
  });

  it("filters by min/max coverage", () => {
    const tiny = rect(0, 0, 2, 2); // 9 cells → 0.0009 < minCoverage
    const plate = rect(2, 2, 98, 98); // ~0.94 > maxCoverage
    const keeper = rect(10, 10, 30, 30); // ~0.044
    const kept = dedupeMaskCandidates([tiny, plate, keeper], {
      minCoverage: 0.004,
      maxCoverage: 0.92,
    });
    expect(kept).toHaveLength(1);
    expect(kept[0]!.bbox.minX).toBe(10);
  });

  it("drops 1-D slivers via the min-side filter", () => {
    const sliver = rect(10, 20, 60, 20); // 51×1 — passes coverage, fails min side
    const keeper = rect(10, 10, 30, 30);
    const kept = dedupeMaskCandidates([sliver, keeper]);
    expect(kept).toHaveLength(1);
    expect(kept[0]!.bbox.minX).toBe(10);
    expect(kept[0]!.bbox.maxY).toBe(30);
  });

  it("drops wide low-aspect edge slivers that clear the min-side check", () => {
    const sliver = rect(5, 20, 85, 24); // 81×5 — side 5 ok, aspect 16 too extreme
    const keeper = rect(10, 40, 40, 70);
    const kept = dedupeMaskCandidates([sliver, keeper], { maxAspect: 8 });
    expect(kept).toHaveLength(1);
    expect(kept[0]!.bbox.minY).toBe(40);
  });

  it("drops the whole-scene mask (bbox fills the frame) but keeps a half-frame food", () => {
    const scene = rect(0, 0, 99, 99); // full 100×100 grid → background/scene mask
    const food = rect(20, 5, 78, 90); // 59×86, interior, ~half the frame
    const kept = dedupeMaskCandidates([scene, food], {
      gridWidth: 100,
      gridHeight: 100,
      maxCoverage: 0.95,
    });
    expect(kept).toHaveLength(1);
    expect(kept[0]!.bbox.minX).toBe(20);
  });

  it("caps the number of returned regions", () => {
    // 20 disjoint 7×7 rects (each above minCoverage) on an 18-px lattice.
    const cands = Array.from({ length: 20 }, (_, i) => {
      const x = (i % 5) * 18;
      const y = Math.floor(i / 5) * 18;
      return rect(x, y, x + 6, y + 6);
    });
    expect(dedupeMaskCandidates(cands, { maxRegions: 5 })).toHaveLength(5);
  });
});
