"""Fine-tune SegFormer on FoodSeg103 for on-device food segmentation.

Why this exists: every public FoodSeg103 checkpoint on Hugging Face measures
at mIoU <= 0.05 (verified 2026-07 — see docs/MODELS.md), while a competent
SegFormer-B0 fine-tune reaches ~0.25 and B1 ~0.32. Shipping quality means
training our own.

GPU job (Google Labs H100): B0 ~2-3 h, B1 ~4-6 h for 60 epochs at batch 32.

    python train/segformer_foodseg103.py \
        --model nvidia/mit-b0 --epochs 60 --batch-size 32 --output out/segformer-b0-food

Export for the app afterwards with export/export_coreml.py (Core ML) or an
ExecuTorch .pte via `fromCustomModel` in react-native-executorch.
"""

from __future__ import annotations

import argparse

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)

DATASET = "EduardoPacheco/FoodSeg103"
NUM_LABELS = 104  # 103 ingredient classes + background


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/mit-b0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--output", default="out/segformer-food")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(DATASET)
    processor = SegformerImageProcessor(do_reduce_labels=False)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        encoded = processor(images, batch["label"], return_tensors="pt")
        return encoded

    train_ds = dataset["train"].with_transform(transform)
    val_ds = dataset["validation"].with_transform(transform)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    metric = evaluate.load("mean_iou")

    @torch.no_grad()
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.from_numpy(logits)
        # Logits come out at 1/4 resolution; upsample to the label size.
        upsampled = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predictions = upsampled.argmax(dim=1).numpy()
        result = metric.compute(
            predictions=predictions,
            references=labels,
            num_labels=NUM_LABELS,
            ignore_index=255,
            reduce_labels=False,
        )
        return {
            "mean_iou": result["mean_iou"],
            "mean_accuracy": result["mean_accuracy"],
            "overall_accuracy": result["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        logging_steps=50,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=8,
        push_to_hub=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output)
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
