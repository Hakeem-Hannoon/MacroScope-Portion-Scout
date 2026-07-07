# expo-portion-capture

The native capture module: a full-screen ARKit camera with a tap-hold-drag metric ruler. Produces the versioned `CapturePayload` consumed by `@ppe/pipeline`.

- Math being implemented: [`../../docs/MATH.md`](../../docs/MATH.md) §2 (raycast ruler), §9 (serialization pitfalls)
- Contract: `@ppe/pipeline` `capturePayloadSchema` — validate every payload there; the native side is an untrusted producer
- Design: [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md)

## Usage

```ts
import * as PortionCapture from "expo-portion-capture";

if (PortionCapture.isSupported()) {
  const payload = await PortionCapture.launch({ requireStroke: true });
  if (payload) {
    const estimate = await estimateMeal(payload, deps); // @ppe/pipeline
  }
}
```

## Requirements

- Expo development build (ARKit is unavailable in Expo Go)
- iOS 16+, ARKit-capable device; LiDAR devices additionally stream scene depth
- `NSCameraUsageDescription` in Info.plist (set via `app.json` → `ios.infoPlist`)

## What the module guarantees

- World coordinates in meters (ARKit VIO — MATH.md §1)
- `camera_to_world` serialized **row-major** (simd is column-major; conversion in `rowMajor(_:)`)
- Intrinsics valid for the stored image resolution; depth intrinsics rescaled to the depth-map resolution
- Capture refuses to fire unless tracking state is `.normal`
- Stroke kind classification: a stroke within ~45° of the plane normal is `vertical` (height), anything else `horizontal` (scale)

## Verification (P0)

Physical-accuracy harness lives in the roadmap: measure 10 known objects at 3 angles, compare stroke lengths to a tape measure. Pass bar: median error ≤ 5 mm on 20 cm spans. Results land in the README results table when run.
