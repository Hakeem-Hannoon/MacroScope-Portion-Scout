# Colab notebooks (the GPU runs)

Run in order on a GPU runtime (H100/A100). Each notebook mounts Drive, clones this repo, and writes every artifact — dataset, checkpoints, manifest, priors, exports — to the shared project Drive folder so disconnects lose nothing:

**Project Drive folder (view-only):** https://drive.google.com/drive/folders/1y1hDjkfHsazsOoe4LJjWKoEKj2bHKKjd

| # | Notebook | GPU | Time | Produces |
|---|---|---|---|---|
| 01 | `01_download_nutrition5k.ipynb` | none | ~1 h (bandwidth) | `data/nutrition5k/` — overhead RGB-D + metadata (~15–25 GB; skips 160 GB of unused video) |
| 02 | `02_train_segformer_foodseg103.ipynb` | H100 | 2–6 h | `checkpoints/segformer-*` + the FoodSeg103 mIoU result row |
| 03 | `03_train_mass_regressor.ipynb` | H100 | ~2 h + extraction | `out/n5k-manifest.csv`, `out/priors.json` (→ update `DEFAULT_KAPPA`), `checkpoints/mass-regressor.pt` + MAPE result row |
| 04 | `04_export_coreml.ipynb` | none | minutes | `out/*.mlpackage.zip` for the iOS app |

Notes:

- **Your data persists to Drive, nothing touches a Google Cloud account.** The Nutrition5k download streams from the public bucket over plain HTTPS (no gcloud, no project, no auth). The Nutrition5k dataset and every **output** — checkpoints, manifest, priors, `.mlpackage`s — live under `DRIVE_ROOT`. Framework **caches** (HuggingFace/torch) stay on the VM's local disk on purpose: the datasets library memory-maps its Arrow files and mmap over Drive's FUSE mount fails; the dataset and the small pretrained backbones re-download in seconds on a fresh VM.
- Set `DRIVE_ROOT` in the first cell of each notebook to the mounted path of the project folder in your Drive.
- **This repo is private, so the clone needs auth.** Add a GitHub token once as a Colab secret named `GH_TOKEN` (🔑 in the left sidebar → "Add new secret" → paste a PAT with `repo` scope → enable notebook access). The clone cells read it automatically and redact it from any output. If you make the repo public instead, no token is needed and the cells still work. The clone now fails **loud** with a clear message if it can't fetch the code — no more silent failure that looks like a training bug.
- **Runtime:** use an A100/H100. Notebook 02 auto-shrinks the batch on a small GPU (T4) so it won't OOM, but a T4 is many hours vs ~2–3 h on an H100.
- 01 and the training notebooks are all resumable — rerun after a disconnect and they skip or resume what's already in Drive.
- When 02/03 finish, paste the printed result rows into the root README's **Testing set & results** table.
