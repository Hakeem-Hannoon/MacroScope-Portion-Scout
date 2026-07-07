# ppe-demo

Minimal Expo app exercising the full stack on a real device: `expo-portion-capture` (AR ruler) → `@ppe/pipeline` (geometry + nutrients) with a placeholder center-square segmenter until the on-device model lands (roadmap P2).

This app lives outside the npm workspaces so the root install stays light; it links the packages by `file:` path.

```bash
cd apps/demo
npm install
npx expo install --fix        # aligns expo/react-native versions with the SDK
npx expo run:ios              # development build on a physical device
```

ARKit needs real hardware — the simulator and Expo Go will report "unsupported". For the P1 validation drill: cook rice, weigh it, capture with a ≥10 cm ruler stroke across the plate, and compare the app's grams against the scale (protocol + pass bars in `docs/ARCHITECTURE.md` §4).
