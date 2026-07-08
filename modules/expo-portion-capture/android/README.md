# Android implementation (ARCore) — v0 implemented

ARCore parity for the iOS module, emitting the identical `CapturePayload` (validated by `@ppe/pipeline`'s `capturePayloadSchema`). ARCore's physical-camera pose uses the same y-up / z-backward convention as ARKit, and `getImageIntrinsics()` matches `acquireCameraImage()` — so `poseFromArkitCameraToWorld` in `@ppe/geometry` consumes both platforms unchanged.

## Pieces

| File | Role |
|---|---|
| `PortionCaptureModule.kt` | Expo module: `isSupported` (ArCoreApk availability), `launch` → activity + promise bridge |
| `ARCaptureActivity.kt` | Session config, tap-hold-drag ruler (`frame.hitTest`), stroke commit/undo, JPEG capture (YUV→NV21), payload builder, UI chrome |
| `BackgroundRenderer.kt` | Camera feed as a fullscreen external-OES quad (`transformCoordinates2d` keeps it aligned) |
| `DisplayRotationHelper.kt` | Feeds display geometry to the session |

Strokes render on a 2D overlay: world anchors are projected to screen space each frame with the display-oriented view/projection matrices — same math, no 3D scene graph dependency.

## v0 scope notes

- `depth: null` — the Depth API (depth-from-motion / ToF) is wired for a later pass; `scale_source` is `"ruler"`.
- Portrait-locked; Play Services for AR install flow handled (`requestInstall`).
- Camera permission requested in-activity; denial cancels the capture cleanly.
