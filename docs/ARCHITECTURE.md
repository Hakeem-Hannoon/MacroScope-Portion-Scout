# Architecture

Two halves: a **native AR capture module** (the ruler — pure geometry, no ML, buildable today) and an **on-device inference pipeline** (segment → classify → portion math → nutrients). This doc specifies both, iOS-first.

```
RN/Expo app
 └── <PortionCapture />  (native module, full-screen)
       ├── ARKit session: plane detection, VIO pose, LiDAR depth if available
       ├── tap-hold-drag ruler (N strokes)
       └── shutter → CapturePayload ──▶ Pipeline (on-device)
                                          ├── 1 segmentation model
                                          ├── 2 classifier
                                          ├── 3 geometry (MATH.md — plain code, no model)
                                          └── 4 SQLite nutrient bundle (FDC + densities)
                                                └──▶ EstimateResult (JSON)
```

## 1. The capture module (`mobile/`)

### Why a custom native module

No maintained RN/Expo library exposes ARKit raycasting + intrinsics + depth capture (react-native-arkit is dead; ViroReact doesn't surface what we need). The [Expo Modules API](https://docs.expo.dev/modules/overview/) makes a Swift module straightforward, and the app already requires a dev build anyway (models, camera). Module name: `expo-portion-capture`.

### iOS implementation (ARKit)

Session config:

```swift
let config = ARWorldTrackingConfiguration()
config.planeDetection = [.horizontal]
if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
    config.sceneReconstruction = .mesh              // LiDAR devices
}
if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
    config.frameSemantics.insert(.smoothedSceneDepth)
}
session.run(config)
```

The ruler gesture (`UILongPressGestureRecognizer`, `minimumPressDuration ≈ 0.15 s`, which also reports drag movement):

```swift
func worldPoint(at screenPoint: CGPoint) -> simd_float3? {
    // Prefer real geometry (LiDAR mesh / mapped plane), fall back to estimated plane.
    let targets: [ARRaycastQuery.Target] = [.existingPlaneGeometry, .estimatedPlane]
    for target in targets {
        if let q = arView.makeRaycastQuery(from: screenPoint, allowing: target, alignment: .any),
           let hit = session.raycast(q).first {
            return simd_make_float3(hit.worldTransform.columns.3)
        }
    }
    return nil
}

// .began   → p1 = worldPoint(at: touch); spawn line node + floating label
// .changed → p2 = worldPoint(at: touch); redraw line; label = fmt(simd_distance(p1, p2))
// .ended   → strokes.append(Stroke(p1: p1, p2: p2, length: simd_distance(p1, p2)))
```

That raycast is exactly `MATH.md §2` (ray from $K^{-1}[u,v,1]^\top$ through the camera pose, intersected with the plane) — ARKit just does it for us, against tracked geometry that improves over the session. `alignment: .any` matters: on LiDAR devices a *vertical* stroke up the side of the food measures its height (`MATH.md §4b`).

Shutter — freeze the current `ARFrame` and export:

```swift
let frame = session.currentFrame!
// frame.capturedImage      → CVPixelBuffer (YCbCr, full sensor res) → HEIC
// frame.camera.intrinsics  → 3×3 K, valid FOR capturedImage's resolution
// frame.camera.transform   → camera-to-world 4×4 (y-up, -z forward — see MATH.md §9.3)
// planeAnchor              → the horizontal plane the strokes landed on (n, d0, extent)
// frame.smoothedSceneDepth → depth map + confidence (LiDAR only)
// frame.camera.trackingState → refuse capture unless .normal
```

UX rules encoded in the module: at least one stroke ≥ 10 cm before the shutter enables (or explicit "skip — lower accuracy" affordance); coach banner while ARKit is initializing ("move your phone slowly over the table"); strokes render with live cm labels; tap a stroke to delete.

### The capture payload

One JSON document + sidecar binaries. This is the contract between capture and pipeline — versioned from day one:

```jsonc
{
  "version": 1,
  "image": "capture.heic",                    // full-res sensor image
  "image_size": [4032, 3024],
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   // for image_size — rescale on resize (MATH.md §9.1)
  "camera_to_world": [/* 4×4 row-major */],
  "plane": { "normal": [0, 1, 0], "d0": -0.412, "extent": [0.9, 0.6] },
  "strokes": [
    { "p1": [x,y,z], "p2": [x,y,z], "length_m": 0.243, "kind": "horizontal" },
    { "p1": [x,y,z], "p2": [x,y,z], "length_m": 0.041, "kind": "vertical" }
  ],
  "depth": { "map": "depth.bin", "confidence": "conf.bin", "size": [256, 192], "intrinsics": [[...]] },  // null without LiDAR
  "tracking": { "state": "normal", "plane_source": "lidar_mesh" },
  "scale_source": "lidar" // "lidar" | "ruler" | "reference_object" | "stated" | "none"  (MATH.md §7)
}
```

### JS API (Expo module)

```ts
import * as PortionCapture from "expo-portion-capture";

const capture = await PortionCapture.launch({ requireStroke: true });
// → CapturePayload | null (user cancelled)
const estimate = await estimateMeal(capture);   // pipeline entry point
```

### Android parity (ARCore)

Same math, different API names: `Session` with `PlaneFindingMode.HORIZONTAL`, `frame.hitTest(x, y)` for the ruler, `frame.getCamera().getImageIntrinsics()`, Depth API (`frame.acquireDepthImage16Bits()`) on supported devices. Raw Depth API confidence maps replace LiDAR confidence. One Kotlin module, identical payload schema. Sequenced after iOS works end-to-end.

### Fallbacks (non-AR captures, shared photos)

The pipeline accepts payloads with `scale_source: "reference_object" | "stated" | "none"`: a detected credit card (85.60 × 53.98 mm) or user-entered plate diameter fills the homography's scale; `none` degrades to prior-based estimation and must be labeled as such in the result. This keeps the library usable on photos that didn't come from the capture screen.

## 2. The inference pipeline

Runs entirely on-device. Stage choices and model IDs live in [`MODELS.md`](MODELS.md); this section fixes the interfaces.

```
CapturePayload
  → preprocess: YCbCr→RGB, orientation normalize, intrinsics rescale check
  → SEGMENT:   image → [{ mask, bbox }]                      (model, ~10–30 MB)
  → CLASSIFY:  crop per mask → { label, topK, confidence }   (model, ~5–20 MB)
  → GEOMETRY:  mask + payload → { A_m2, h_m?, V_ml, method } (plain code — MATH.md §3–4)
  → RESOLVE:   label → density ρ + per-100g nutrients        (SQLite bundle)
  → COMPOSE:   m = ρV → kcal/macros/micros + Atwater check + confidence
```

Geometry is deliberately **not** a model: it's ~300 lines of linear algebra (homography build/apply, shoelace area, height-field integration) that run in microseconds and are unit-testable against synthetic scenes with known ground truth. Model inference happens once per capture (not per frame), so latency budgets are generous (< 2 s total on an A16).

### Result contract

Mirrors the shape a logging app wants (deliberately compatible with Spotter's `MealItem`):

```jsonc
{
  "items": [{
    "label": "white rice, cooked",
    "confidence": 0.86,
    "geometry": { "area_cm2": 118, "height_cm": 3.1, "volume_ml": 240, "method": "ruler+height" },
    "mass_g": 161,
    "kcal": 209,
    "protein_g": 4.3, "carbs_g": 45.0, "fat_g": 0.5,
    "micros": { "fiberG": 0.6, "sodiumMg": 2, "potassiumMg": 55, "..." : 0 },
    "flags": []                       // e.g. "atwater_mismatch", "container_prior_used"
  }],
  "totals": { "kcal": 640, "protein_g": 38, "carbs_g": 71, "fat_g": 22, "micros": {} },
  "quality": {
    "scale_source": "ruler",
    "ruler_residual_mm": 2.1,         // MATH.md §3.1 self-check
    "est_relative_error": 0.22        // from the MATH.md §8 budget, given the sources used
  }
}
```

`est_relative_error` is surfaced to the UI so portions render as ranges ("~160 g ± 35 g"), never false precision — and everything stays user-editable before logging.

## 3. Repo mapping

| Dir | Contents |
|---|---|
| `packages/geometry` | the math library (TypeScript, zero deps) — MATH.md as tested code |
| `packages/pipeline` | zod contracts, model adapters, `estimateMeal` orchestration, mocks |
| `modules/expo-portion-capture` | the native capture module (Swift/ARKit implemented; Kotlin/ARCore planned) |
| `apps/demo` | Expo dev-build app exercising capture → pipeline on-device |
| `model/` | training (SegFormer FoodSeg103, scale-conditioned regressor), prior fitting, Core ML export |
| `nutrition/` | ETL: USDA FDC + FNDDS portions + FAO/INFOODS densities → versioned SQLite bundle (~15k generic foods) |

## 4. Testing strategy

- **Geometry unit tests**: synthetic camera + plane + known-size quads → homography must recover lengths/areas to float precision; §9 pitfalls each get a regression test (intrinsics rescale, orientation, y-flip).
- **P0 physical test**: ruler vs tape measure, 10 objects × 3 angles ×2 lighting; pass = median error ≤ 5 mm on 20 cm spans.
- **P1 physical test**: geometry-only pipeline (hand mask + hand label) vs kitchen scale on ~30 home meals; pass = median mass error ≤ 25% (v0 budget from MATH.md §8).
- **P3 benchmark**: Nutrition5k test split, report calorie MAPE vs the published 26.1%/16.5% baselines.
