import type {
  Classifier,
  ClassifierResult,
  FoodRecord,
  NutrientStore,
  Region,
  Segmenter,
} from "./adapters";

/** Test/demo segmenter that returns fixed regions. */
export class FixedSegmenter implements Segmenter {
  constructor(private readonly regions: Region[]) {}
  segment(): Promise<Region[]> {
    return Promise.resolve(this.regions);
  }
}

/** Test/demo classifier that labels every region the same way. */
export class FixedClassifier implements Classifier {
  constructor(private readonly result: ClassifierResult) {}
  classify(): Promise<ClassifierResult> {
    return Promise.resolve(this.result);
  }
}

/** In-memory nutrient store for tests and the demo app. */
export class InMemoryNutrientStore implements NutrientStore {
  private readonly byLabel = new Map<string, FoodRecord>();
  constructor(records: FoodRecord[]) {
    for (const r of records) this.byLabel.set(r.label.toLowerCase(), r);
  }
  lookup(label: string): Promise<FoodRecord | null> {
    return Promise.resolve(this.byLabel.get(label.toLowerCase()) ?? null);
  }
}
