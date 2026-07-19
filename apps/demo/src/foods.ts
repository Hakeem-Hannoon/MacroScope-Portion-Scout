import type { Classifier, ClassifierResult, Region } from "@ppe/pipeline";
// The canonical label→FDC map (nutrition/label-map.json), copied next to the DB
// asset by `npm run build:nutrients`. See docs/REAL_ADAPTERS.md §1.
import labelMap from "../assets/label-map.json";
// The on-device classifier vocabulary (assets/food-vocab-embeddings.json) — the
// common FoodSeg103 subset. Its terse labels resolve to FDC rows via labelMap.
import vocabDoc from "../assets/food-vocab-embeddings.json";

/**
 * Every food the classifier can name — the terse vocabulary labels, used for the
 * propose→confirm correction chips so the user can relabel a region to any known
 * food (each resolves to REAL USDA nutrition via labelMap + the starter bundle).
 * Kept in sync with the model automatically: it's the same vocabulary the image
 * encoder is matched against (nutrition/starter/foods.mjs is the source of both).
 */
export const STARTER_FOODS: string[] = (
  vocabDoc as { vocab: { label: string }[] }
).vocab.map((v) => v.label);

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
