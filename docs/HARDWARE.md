# Hardware

What each sensor contributes, which devices provide it, and the compute budget for the on-device models.

## The sensors behind the math

**IMU (accelerometer + gyroscope).** The foundation of the whole idea. The accelerometer measures specific force in m/s² at hundreds of Hz; ARKit/ARCore fuse it with camera tracking (visual-inertial odometry) so world coordinates come out in meters (MATH.md §1). Every ARKit/ARCore phone has one, which is why the ruler works on hardware without any depth sensor. Practical accuracy after a second of gentle device motion: roughly ±0.5–1 cm on tabletop-scale distances — P0 measures this exactly.

**Camera + factory calibration.** `frame.camera.intrinsics` carries the per-device calibrated pinhole parameters (fx, fy, cx, cy) at the captured resolution — the K matrix in every equation of MATH.md. Two handling rules: rescale K with any image resize (§9.1), and apply orientation exactly once (§9.2). Optical image stabilization perturbs intrinsics by fractions of a pixel — ignorable at food distances.

**LiDAR scanner (iPhone/iPad Pro models, iPhone 12 Pro onward).** A sparse time-of-flight emitter (~576 points) that ARKit fuses with the RGB stream into a dense 256×192 depth map at up to 60 Hz (`sceneDepth`/`smoothedSceneDepth`), plus a per-pixel confidence map. Contribution: direct metric height fields (MATH.md §4a — the highest-accuracy volume route), raycasts against arbitrary surfaces (height strokes up the side of the food), and instant plane detection. Resolution caveat: 256×192 across a ~60° FOV puts roughly 3–5 mm between depth samples at 40 cm — fine for portions, marginal for garnish.

**Android depth.** ARCore's Depth API produces comparable depth maps from motion (any certified device) or from hardware ToF sensors where present. Coverage is fragmented; the capture module treats depth as an enhancement everywhere and requires only hit-testing.

## Device tiers

| Tier | Hardware | Scale source | Expected per-item error (MATH.md §8) |
|---|---|---|---|
| 1 | iPhone/iPad Pro with LiDAR | depth + ruler verification | ~15–20% |
| 2 | Any ARKit device (iPhone SE 2 and up) / ARCore device | VIO ruler | ~20–30% |
| 3 | Camera only (shared photo, web upload) | reference object or stated plate size | ~30%+ |
| 4 | No scale information | priors only | labeled estimate |

## Requirements

- **iOS:** iOS 16+, A12 Bionic or newer for solid ARKit tracking (the podspec pins 16.0). Camera permission string required.
- **Android (planned):** ARCore-certified device with Google Play Services for AR; Depth API where supported.
- **Expo:** development builds only — ARKit and the model runtimes are native modules that Expo Go excludes.

## Compute budget for the models

Published on-device numbers for the exact artifacts we use (docs/MODELS.md):

| Model | Size | Latency |
|---|---|---|
| MobileCLIP-S0 image encoder (Core ML) | ~11 M params | 1.5 ms (iPhone) |
| Depth-Anything-V2-small (Core ML fp16) | 49.8 MB | 31–34 ms (iPhone 12/15 Pro Max) |
| SAM 2.1 Tiny (Core ML fp16) | ~39 M params | interactive-rate; benchmark on target devices |
| SegFormer-B0 fine-tune (ours) | 3.7 M params | benchmark after export |
| Mass regressor (ours, MobileNetV3 + FiLM) | ~5.5 M params | expect ≤ 10 ms on ANE |

Inference runs once per capture (single frame, no live loop), so the entire model stack fits comfortably under a 2-second budget on an A15+, with the Apple Neural Engine doing the heavy lifting via Core ML. Real-time hover-scanning (MFP Meal Scan style) would change the budget completely; the roadmap keeps it out of scope.

**Memory/asset footprint:** models ~60–120 MB total on disk depending on the chosen segmentation route, nutrient SQLite bundle ~15–30 MB (generic foods), capture payloads ~2–4 MB each (HEIC + optional 200 KB depth buffer).
