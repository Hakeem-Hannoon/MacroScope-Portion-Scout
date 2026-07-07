# Models

Verified inventory of off-the-shelf models per pipeline stage (web-verified 2026-07-07; ✓ = model page fetched directly, ~ = from listings/search only). Strategy: ship the smallest working artifact per stage, fine-tune only where public checkpoints fail, and train from scratch exactly once — the scale-conditioned regressor, where nothing public exists.

## 1. Segmentation

**Use as-is:** [`apple/coreml-sam2.1-tiny`](https://huggingface.co/apple/coreml-sam2.1-tiny) ✓ — SAM 2.1 Tiny as fp16 Core ML packages, Apache-2.0, promptable with points/boxes. The ruler gesture already puts the user's finger on the food; the same tap becomes the segmentation prompt. Demo app: [SAM2 Studio](https://github.com/huggingface/sam2-studio). Alternatives: [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) ✓ (Apache-2.0, 9.66 M params, ~12 ms on GPU, official ONNX export) and [EfficientSAM](https://github.com/yformer/EfficientSAM) ✓ (Apache-2.0).

**Fine-tune (the automatic path):** SegFormer-B0/B1 (`nvidia/mit-b0`/`b1`) on [`EduardoPacheco/FoodSeg103`](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) ✓ (Apache-2.0, 104 classes, 7.1k images). **Warning, verified:** every public FoodSeg103 checkpoint on HF measures at mIoU ≤ 0.05 — e.g. [`LightDestory/segformer-b0-…`](https://huggingface.co/LightDestory/segformer-b0-finetuned-segments-food-oct-24v2) ✓ at mIoU 0.0104. Ship none of them. Competent fine-tunes reach ~0.25 (B0) / ~0.32 (B1) ~; the published ceiling is FoodSAM at 46.4 mIoU ✓ (server-class). Training script: `model/train/segformer_foodseg103.py`.

**License flags:** all Ultralytics-lineage weights (YOLOv8/11/26-seg, FastSAM) are AGPL-3.0 ✓ — poison for a closed-source app without a paid license; community "apache" tags on YOLO fine-tunes are legally dubious. Apache-clean detector alternative: [RF-DETR](https://github.com/roboflow/rf-detr) seg variants ✓ (COCO classes; food appears coarsely). FoodSeg103's images derive from Recipe1M — training-data provenance carries a wrinkle worth remembering ~.

**React Native path:** [react-native-executorch](https://github.com/software-mansion/react-native-executorch) ✓ ships `rfdetr-nano-seg`, `fastsam-s/x` (AGPL upstream ~), DeepLab-v3 (Pascal VOC — wrong classes), and custom `.pte` loading (`fromCustomModel`, v0.8+) — our SegFormer fine-tune rides that on Android; Core ML on iOS.

## 2. Classification

**Use as-is:** [`apple/coreml-mobileclip`](https://huggingface.co/apple/coreml-mobileclip) ✓ — MobileCLIP S0–B Core ML encoders (image 1.5–10.4 ms on iPhone), license `apple-ascl` (custom; review before shipping). Zero-shot recipe: embed each segmented crop, cosine-match against precomputed text embeddings of the food vocabulary (FoodSeg103 labels + USDA FDC descriptions, prompt-ensembled). Text side is computed offline; runtime cost is the image encoder only. CLIP-class zero-shot lands ~88–90% top-1 on Food-101 ~. Quick POC alternative: [`nateraw/food`](https://huggingface.co/nateraw/food) ✓ — ViT-B/16, 89.13% Food-101, Apache-2.0.

**Fine-tune:** [`apple/MobileCLIP2-S0`](https://huggingface.co/apple/MobileCLIP2-S0) ✓ (11.4 M image params, 71.5% zero-shot IN-1k, `apple-amlr`) with a linear head, or an EfficientNet/FastViT head for the ExecuTorch path. The gap worth a small head: cooked-state disambiguation (fried vs steamed) — it drives density (MATH.md §5).

## 3. Depth fallback (devices without LiDAR)

**Use as-is:** [`apple/coreml-depth-anything-v2-small`](https://huggingface.co/apple/coreml-depth-anything-v2-small) ✓ — Apache-2.0, 24.8 M params, 49.8 MB fp16, **31–34 ms on iPhone 12/15 Pro Max** ✓. Output is *relative* inverse depth; the ruler stroke supplies the metric anchor to rescale it — which is precisely the trick this project is built on, so a metric-depth network stays optional.

Metric alternatives if ever needed: [`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) ✓ (24.8 M, license unstated on card ~), DA3METRIC-LARGE ✓ (Apache-2.0, 0.35 B — heavy). Ruled out: Apple DepthPro Core ML ✓ (1.9 GB, seconds-class), UniDepth v2 ✓ and DA-V2 Base/Large ✓ (CC-BY-NC). MiDaS is archived ✓.

## 4. Direct image→nutrition (the gap we fill)

Google released no Nutrition5k checkpoints ✓. Public attempts are either server-scale ([`Yueha0/FoodLMM-Chat`](https://github.com/YuehaoYin/FoodLMM) ✓ — 7B + SAM-H) or experiments ([`jc-builds/CalorieCLIP`](https://huggingface.co/jc-builds/CalorieCLIP) ✓ — MIT, calories only, self-reported 51 kcal MAE; assorted student CNNs ~). Depth-conditioned regression exists in papers only (DPF-Nutrition, FLAVA RGB-D at ~14.4% PMAE — no weights ✓).

**Verified conclusion: no public model conditions on a measured AR scale reference. Nobody has shipped this.** That model is `model/train/mass_regressor_nutrition5k.py`.

### v2 regressor architecture (the CNN/RNN/anything decision)

- **Backbone: small CNN or CNN-ViT hybrid** — `mobilenetv3_large_100` default, `fastvit_t8` for ANE latency. Single-image visual encoding is convolution/attention territory, and pretrained small backbones transfer well at Nutrition5k's size (~3.5k usable overhead dishes; a from-scratch ViT would starve).
- **Physics conditioning via FiLM**: the measured scalars (log metric area, height, scale source) generate per-channel (γ, β) that modulate the visual embedding. This mirrors the physics — mass scales with measured geometry for fixed appearance — and beats naive concatenation, which lets the network ignore the scalars early in training.
- **Heads**: log-mass + auxiliary log-kcal, SmoothL1 in log space (optimizes relative error, matching the %MAPE benchmark).
- **RNNs: no.** Recurrence models sequences; a capture is one frame plus scalars. If multi-frame sweeps land later, attention pooling over per-frame embeddings covers it with the same backbone.
- **Targets**: beat 26.1% calorie MAPE (RGB baseline); approach 16.5% (depth baseline). The regressor sees strictly more information than the RGB baseline — the measured scale — so landing between them is the expected outcome, with the geometry-only pipeline (v0/v1) as the fallback whenever the model's confidence is low.
- Export: `model/export/export_coreml.py` (Core ML fp16); ExecuTorch `.pte` for Android via the same traced graph.

## 5. Data

- [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) ✓ — CC BY 4.0 (commercial-clean), 181.4 GB via `gsutil`, overhead RGB-D at 0.1 mm depth units, per-ingredient masses, official splits. The training substrate for the regressor and the priors.
- [FAO/INFOODS Density Database v2](https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/) ✓ — 638 foods with g/mL; the primary density source. FDC/FNDDS portion weights are the secondary source, with USDA's own caveat that portion weights approximate density ✓ (`nutrition/` tags every derived density with its source).
- [USDA FoodData Central](https://fdc.nal.usda.gov/) — CC0 nutrient values (see `nutrition/`).

## Open items

Unverified from the sweep: exact quality of the `tanganke/*_food101` CLIP fine-tunes; ML-Kit-GenAI image support on Android; the Metric-Indoor-Small license field; MobileCLIP2 Core ML availability (v1 only today ✓ — convert v2 via coremltools if needed). Apple's `apple-ascl`/`apple-amlr` custom licenses want a read before App Store submission.
