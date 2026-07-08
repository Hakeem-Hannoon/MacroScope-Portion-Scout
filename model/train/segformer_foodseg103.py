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


def mean_iou_metrics(preds, labels, num_labels: int, ignore_index: int = 255) -> dict:
    """Mean IoU + accuracy from integer label maps, computed directly in NumPy.

    We deliberately do NOT use `evaluate.load("mean_iou")`: that metric is
    fetched from the HF Hub at runtime and its current version validates inputs
    as `Image` features, so it rejects the raw (H, W) class arrays a segmenter
    produces — the run dies with "Predictions and/or references don't match the
    expected format". A confusion matrix over all pixels is exact, dependency-
    free, and version-stable:

        IoU_c = TP_c / (TP_c + FP_c + FN_c)

    averaged over the classes that actually appear (absent classes are NaN and
    dropped by nanmean — matching mean_iou's semantics). Unlabeled pixels
    (== ignore_index) are excluded first.
    """
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_labels)
    preds, labels = preds[valid], labels[valid]
    # One confusion matrix from the (true, pred) pixel pairs via a single bincount.
    conf = np.bincount(
        num_labels * labels.astype(np.int64) + preds.astype(np.int64),
        minlength=num_labels ** 2,
    ).reshape(num_labels, num_labels)
    tp = np.diag(conf).astype(np.float64)
    per_true = conf.sum(axis=1)  # ground-truth pixels per class (TP + FN)
    per_pred = conf.sum(axis=0)  # predicted  pixels per class (TP + FP)
    union = per_true + per_pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(per_true > 0, tp / per_true, np.nan)
    total = conf.sum()
    return {
        "mean_iou": float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0,
        "mean_accuracy": float(np.nanmean(acc)) if np.isfinite(acc).any() else 0.0,
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/mit-b0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--output", default="out/segformer-food")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    # 1. Data. FoodSeg103 streams from the Hub. The processor normalizes the
    #    images and aligns the label maps; do_reduce_labels=False keeps class 0
    #    as background (NUM_LABELS already counts it). with_transform runs this
    #    lazily per batch, so the whole set never decodes up front.
    dataset = load_dataset(DATASET)
    processor = SegformerImageProcessor(do_reduce_labels=False)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        encoded = processor(images, batch["label"], return_tensors="pt")
        return encoded

    train_ds = dataset["train"].with_transform(transform)
    val_ds = dataset["validation"].with_transform(transform)

    # 2. Model. Load the pretrained MiT backbone but swap in a fresh head sized
    #    to our 104 classes — ignore_mismatched_sizes drops the incompatible
    #    pretrained head instead of erroring on the shape mismatch.
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    # 3. Metric: mean IoU across all classes — the exact number every public
    #    checkpoint fails at. Computed in NumPy (mean_iou_metrics above), not via
    #    evaluate.load("mean_iou") whose Hub version rejects raw label arrays.
    #    The two helpers below keep the computation from OOMing on the full set.
    def preprocess_logits_for_metrics(logits, labels):
        # Collapse (B, C, h, w) → (B, h, w) by arg-maxing on-device BEFORE the
        # Trainer accumulates predictions across the whole val set. Without
        # this the run holds every logit (104 channels × 2,140 images) in RAM
        # and OOMs during the FIRST epoch's eval — which runs just before the
        # first checkpoint save, so nothing is ever written (OUT stays empty).
        return logits.argmax(dim=1)

    @torch.no_grad()
    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds: (B, h, w) at model output resolution
        # Upsample the small integer prediction map (nearest) to label size —
        # cheap, and it never materializes the per-class logit volume.
        preds = torch.nn.functional.interpolate(
            torch.from_numpy(preds).unsqueeze(1).float(),
            size=labels.shape[-2:],
            mode="nearest",
        ).squeeze(1).long().numpy()
        return mean_iou_metrics(preds, labels, NUM_LABELS, ignore_index=255)

    # 4. Training config. Evaluate and checkpoint every epoch, keep the
    #    best-by-mIoU model at the end (save_total_limit=2 prunes the rest),
    #    fp16 on GPU for throughput.
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

    # 5. Train, then persist. save_model writes the weights; the metric files
    #    below are written separately so the result row never depends on a
    #    checkpoint dir that Drive's FUSE layer may not have synced.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(args.output)
    # Persist the metric next to the model so reporting never depends on
    # checkpoint dirs surviving (Google Drive's FUSE layer may not sync them,
    # and save_total_limit prunes them). save_metrics writes eval_results.json;
    # save_state writes trainer_state.json (log_history + best_metric).
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    trainer.save_state()
    print(metrics)


if __name__ == "__main__":
    main()
