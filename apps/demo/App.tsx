import { useState } from "react";
import {
  Button,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import * as PortionCapture from "expo-portion-capture";
import {
  type EstimateResult,
  type Region,
  type Segmenter,
  FixedClassifier,
  InMemoryNutrientStore,
  estimateMeal,
} from "@ppe/pipeline";

/**
 * Placeholder segmenter until the on-device model lands (roadmap P2): a
 * centered square covering ~16% of the frame. Combined with a real capture
 * this exercises the full metric path — ruler → homography → area → volume
 * → mass → nutrients — against food you can weigh on a kitchen scale (P1).
 */
class CenterSquareSegmenter implements Segmenter {
  segment(_imageUri: string, [w, h]: [number, number]): Promise<Region[]> {
    const side = Math.min(w, h) * 0.4;
    const cx = w / 2;
    const cy = h / 2;
    return Promise.resolve([
      {
        polygonPx: [
          [cx - side / 2, cy - side / 2],
          [cx + side / 2, cy - side / 2],
          [cx + side / 2, cy + side / 2],
          [cx - side / 2, cy + side / 2],
        ] as [number, number][],
      },
    ]);
  }
}

const deps = {
  segmenter: new CenterSquareSegmenter(),
  classifier: new FixedClassifier({ label: "white rice, cooked", confidence: 0.99 }),
  nutrients: new InMemoryNutrientStore([
    {
      label: "white rice, cooked",
      per100: {
        kcal: 130,
        proteinG: 2.7,
        carbsG: 28.2,
        fatG: 0.3,
        micros: { potassiumMg: 35, sodiumMg: 1 },
      },
      densityGPerMl: 0.67,
      shape: { kind: "mound", kappa: 0.55 },
    },
  ]),
};

export default function App() {
  const [result, setResult] = useState<EstimateResult | null>(null);
  const [status, setStatus] = useState("Ready");

  const capture = async () => {
    if (!PortionCapture.isSupported()) {
      setStatus("ARKit is unavailable — use a development build on a real device");
      return;
    }
    setStatus("Capturing…");
    try {
      const payload = await PortionCapture.launch({ requireStroke: true });
      if (!payload) {
        setStatus("Cancelled");
        return;
      }
      setStatus("Estimating…");
      setResult(await estimateMeal(payload, deps));
      setStatus("Done — placeholder segmentation, real geometry");
    } catch (error) {
      setStatus(`Failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />
      <Text style={styles.title}>Physics-Powered Portion Estimation</Text>
      <Text style={styles.subtitle}>{status}</Text>
      <Button title="Capture a meal" onPress={capture} />
      {result && (
        <ScrollView style={styles.results}>
          {result.items.map((item, i) => (
            <View key={i} style={styles.card}>
              <Text style={styles.label}>{item.label}</Text>
              <Text>
                {item.geometry.area_cm2} cm² · {item.geometry.volume_ml} mL ·{" "}
                {item.geometry.method}
              </Text>
              <Text>
                {item.mass_g ?? "?"} g · {item.kcal ?? "?"} kcal · P{" "}
                {item.protein_g ?? "?"} / C {item.carbs_g ?? "?"} / F {item.fat_g ?? "?"}
              </Text>
              {item.flags.length > 0 && <Text>flags: {item.flags.join(", ")}</Text>}
            </View>
          ))}
          <View style={styles.card}>
            <Text style={styles.label}>Quality</Text>
            <Text>scale: {result.quality.scale_source}</Text>
            <Text>ruler residual: {result.quality.ruler_residual_mm ?? "—"} mm</Text>
            <Text>
              estimated error: ±{Math.round(result.quality.est_relative_error * 100)}%
            </Text>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, paddingTop: 80, paddingHorizontal: 20, gap: 12 },
  title: { fontSize: 20, fontWeight: "600" },
  subtitle: { color: "#666" },
  results: { marginTop: 16 },
  card: {
    padding: 12,
    borderRadius: 8,
    backgroundColor: "#f2f2f2",
    marginBottom: 10,
    gap: 4,
  },
  label: { fontWeight: "600" },
});
