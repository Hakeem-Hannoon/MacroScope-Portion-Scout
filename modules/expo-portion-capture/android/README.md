# Android implementation — planned

ARCore parity for the iOS module, same payload contract (`@ppe/pipeline` `capturePayloadSchema`). The math is identical; the API names differ:

| iOS (ARKit) | Android (ARCore) |
|---|---|
| `ARRaycastQuery` / `session.raycast` | `frame.hitTest(x, y)` |
| `frame.camera.intrinsics` | `camera.getImageIntrinsics()` |
| `frame.camera.transform` | `camera.getPose()` (convert to row-major camera-to-world) |
| `ARPlaneAnchor` | `Plane` trackable |
| `smoothedSceneDepth` (LiDAR) | Depth API `frame.acquireDepthImage16Bits()` (supported devices) |

Sequenced after the iOS module is validated end-to-end (P0/P1 in the roadmap). Device requirements and depth coverage: `docs/HARDWARE.md`.
