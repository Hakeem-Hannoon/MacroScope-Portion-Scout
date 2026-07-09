import type { Classifier, ClassifierResult, Region } from "@ppe/pipeline";
// The canonical label→FDC map (nutrition/label-map.json), copied next to the DB
// asset by `npm run build:nutrients`. See docs/REAL_ADAPTERS.md §1.
import labelMap from "../assets/label-map.json";

/**
 * The foods in the starter bundle (apps/demo/assets/nutrients.sqlite). Until the
 * on-device classifier (MobileCLIP) is wired — see vision-adapters.ts — the demo
 * lets the user pick which of these the plate is, so the estimate uses REAL USDA
 * nutrition for the REAL measured portion. These strings are the exact bundle
 * descriptions, so the store resolves them by exact match.
 */
export const STARTER_FOODS: string[] = [
  "Rice, white, cooked",
  "Chicken breast, cooked, roasted",
  "Broccoli, cooked, boiled",
  "Egg, whole, cooked, hard-boiled",
  "Salmon, Atlantic, cooked",
  "Pasta, cooked, enriched",
  "Potato, baked, flesh and skin",
  "Ground beef, 85% lean, cooked",
  "Banana, raw",
  "Apple, raw, with skin",
  "Almonds, raw",
  "Bread, white, commercial",
];

/**
 * Terse classifier label → bundle description. A real MobileCLIP head emits short
 * labels ("rice", "chicken"); this is the curated map the NutrientStore uses to
 * resolve them (STATUS.md's "quality-critical data artifact"), loaded from the
 * shared `nutrition/label-map.json`. `_`-prefixed keys (e.g. `_comment`) are meta.
 */
export const FOOD_ALIASES: Record<string, string> = Object.fromEntries(
  Object.entries(labelMap as Record<string, string>).filter(([k]) => !k.startsWith("_")),
);

/**
 * A `Classifier` whose answer is whatever the UI currently has selected — the
 * interim stand-in for the on-device model. Real geometry + real nutrition; the
 * label is user-confirmed rather than predicted. Swap for `ZeroShotClipClassifier`
 * (vision-adapters.ts) once the MobileCLIP model is bundled.
 */
export class SelectedClassifier implements Classifier {
  constructor(private label: string) {}
  setLabel(label: string): void {
    this.label = label;
  }
  classify(_imageUri: string, _region: Region): Promise<ClassifierResult> {
    return Promise.resolve({ label: this.label, confidence: 1 });
  }
}
