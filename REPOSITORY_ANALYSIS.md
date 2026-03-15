# Complete Repository Analysis: Anomalous Sound Detection Baseline

This document explains the entire repository in a structured, beginner-friendly way. It connects everything to your thesis goal: **detecting abnormal sounds in a toy RC car** using anomaly detection.

---

# Part 1: Overall System Pipeline (Start to Finish)

## 1.1 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        ANOMALOUS SOUND DETECTION PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
  │   DATASET    │     │  PREPROCESSING  │     │ FEATURE          │     │   MODEL     │
  │              │     │  (load WAV)     │     │ EXTRACTION       │     │ (AutoEncoder)│
  │ • WAV files  │ ──► │ • Raw waveform  │ ──► │ • Log-mel        │ ──► │ • Encode    │
  │ • Normal only│     │ • Mono, sr      │     │ • Frame concat   │     │ • Decode    │
  │   for train  │     │   (librosa)     │     │ • Vectors (640D) │     │ • Reconstruct│
  └──────────────┘     └─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                                               │
                                                                               ▼
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
  │  EVALUATION  │     │  DECISION       │     │ ANOMALY SCORE     │     │ Reconstruction│
  │              │ ◄── │ • Normal/Anomaly │ ◄── │ • MSE or         │ ◄──  │ error        │
  │ • AUC, pAUC  │     │ • score vs       │     │   Mahalanobis    │     │ (input vs    │
  │ • Precision  │     │   threshold      │     │ • One number     │     │  output)     │
  │ • Recall, F1 │     │ • 0 or 1         │     │   per file       │     │              │
  └──────────────┘     └─────────────────┘     └──────────────────┘     └─────────────┘
```

**In one sentence:** Audio files are turned into log-mel feature vectors, an AutoEncoder learns to reconstruct “normal” sound, and the **reconstruction error** becomes the **anomaly score**; high score → anomaly.

**For your toy RC car:** Normal = healthy motor/gearbox; anomaly = unusual rattling, grinding, or whine. The pipeline is the same; only the data (and later, possibly the scoring method) change.

---

## 1.2 Step-by-Step (No Code Yet)

| Step | What happens | Why it matters for RC car |
|------|----------------|---------------------------|
| **1. Dataset** | You choose a machine type (e.g. ToyCar). Training uses only **normal** WAVs from that type. | You will use normal RC car recordings only for training. |
| **2. Preprocessing** | Each WAV is loaded as a waveform (samples), optionally converted to mono. | Same for RC car: load your recordings. |
| **3. Feature extraction** | Waveform → mel spectrogram → log → sliding windows of 5 time frames × 128 mel bins → **640-dimensional vectors**. | Converts “raw sound” into a format the model can learn from. Same process for RC car. |
| **4. Model** | AutoEncoder: 640 → 128 → 128 → 128 → 128 → **8** (bottleneck) → 128 → … → 640. It learns to copy normal input. | Learns a compact “normal sound” representation. |
| **5. Anomaly score** | For each input vector, compare **output** to **input**. Average squared difference (MSE) or Mahalanobis distance → one score per file. | High error = sound didn’t match “normal” = likely anomaly (e.g. fault). |
| **6. Decision** | Compare score to a **threshold**. Above = anomaly (1), below = normal (0). Threshold is set from the distribution of normal scores. | Lets you say “this recording is normal” or “this is anomalous.” |
| **7. Evaluation** | With labels (dev set): compute AUC, pAUC, precision, recall, F1; plot score distributions and loss curves. | Tells you how well the system separates normal vs anomaly. |

---

# Part 2: Data Flow Inside the Code

## 2.1 Where Audio Files Are Loaded

**File:** `datasets/loader_common.py`

- **Function:** `file_load(wav_name, mono=False)`  
  - Uses **librosa.load(wav_name, sr=None, mono=mono)**.  
  - **Returns:** `(y, sr)` — waveform array `y` and sample rate `sr`.  
  - So: **audio files are first loaded here** as raw samples.

**Who calls it:**  
`file_to_vectors(file_name, ...)` in the same file calls `file_load(file_name, mono=True)` for each WAV. So **all loading goes through** `file_load` → `file_to_vectors`.

**Path in your run:**  
Dataset class (`DCASE202XT2Loader` in `datasets/dcase_dcase202x_t2_loader.py`) gets a list of WAV paths from `file_list_generator` (in `loader_common.py`). For each path it eventually calls `file_list_to_data` → `file_to_vectors` → `file_load`. So:

- **Audio files are loaded in:** `datasets/loader_common.py` inside `file_to_vectors()` via `file_load()`.

---

## 2.2 How Audio Is Transformed (Features)

**File:** `datasets/loader_common.py`  
**Function:** `file_to_vectors(file_name, n_mels, n_frames, n_fft, hop_length, power, ...)`

**Step-by-step transformation:**

1. **Load WAV:** `y, sr = file_load(file_name, mono=True)` → time-domain signal.
2. **Mel spectrogram:**  
   `librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128, ...)`  
   → 2D array shape `(n_mels, time_frames)` — frequency (mel) × time.
3. **Log:**  
   `20.0 / power * np.log10(max(mel_spectrogram, tiny))`  
   → log-mel spectrogram (more like how we hear loudness).
4. **Framing:**  
   Sliding window of `n_frames` (e.g. 5) consecutive time frames. Each window is flattened:  
   `dims = n_mels * n_frames` (e.g. 128×5 = 640).  
   So each **vector** = 640 numbers (one “snippet” of the sound).
5. **Output:**  
   NumPy array of shape `(n_vectors, 640)` — many 640-D vectors per file.

So: **audio is transformed in** `datasets/loader_common.py` in **`file_to_vectors()`**: WAV → mel spectrogram → log → framed vectors.

**Data type:** Float NumPy array. Later this is wrapped in PyTorch tensors by the dataset/DataLoader.

---

## 2.3 Data Format Between Modules

| Stage | Format | Shape example | Where |
|-------|--------|----------------|-------|
| After `file_to_vectors` | NumPy float | `(n_vectors, 640)` per file | `loader_common.py` |
| After `file_list_to_data` | NumPy float | `(total_vectors, 640)` for many files | `dcase_dcase202x_t2_loader.py` |
| In DataLoader batch | PyTorch tensor | `(batch_size, 640)` | `dcase2023t2_ae.py` train/eval |
| Model input | PyTorch tensor | `(B, 640)` or `(B, 1, 5, 128)` viewed as `(B, 640)` | `network.py` `forward()` |
| Model output | PyTorch tensor | `recon`: `(B, 640)`, `z`: `(B, 8)` | `network.py` `forward()` |

So: **the model receives** a batch of **640-dimensional vectors** (float tensor). It **outputs** the **reconstructed 640-D vector** and the **8-D latent (bottleneck)**.

---

## 2.4 What the Model Receives and Outputs

**File:** `networks/dcase2023t2_ae/network.py`  
**Class:** `AENet`

- **Input:** `x` — shape `(batch_size, input_dim)` where `input_dim = 640` (e.g. 5×128×1). In `forward`: `x.view(-1, self.input_dim)` so it’s always `(B, 640)`.
- **Output:**  
  - `recon_x` = decoder(encoder(x)) — same shape as input, **(B, 640)**.  
  - `z` = bottleneck activations — **(B, 8)**.

So: **model input** = batch of 640-D feature vectors; **model output** = 640-D reconstruction + 8-D latent. The **anomaly score** is computed from **input vs reconstruction** (and optionally covariance of their difference for Mahalanobis), not from `z` directly in the default MSE setup.

---

# Part 3: Role of Each Major File

## 3.1 Entry and Configuration

| File | Role |
|------|------|
| **train.py** | Entry point. Loads config from YAML, parses args, creates the model class, runs training loop (epochs) and then test if not `--test_only`. |
| **common.py** | Defines `yaml_load()` (reads `baseline.yaml`), `get_argparse()` (all CLI and YAML-mapped options: dataset, model, score type, feature params, training params, paths). |
| **baseline.yaml** | Default config: dataset path, model name, **score** (MSE or MAHALA), feature (n_mels, frames, n_fft, hop_length), training (epochs, lr, batch_size), decision_threshold, etc. |

## 3.2 Dataset and Feature Extraction

| File | Role |
|------|------|
| **datasets/datasets.py** | Defines which dataset class to use (e.g. DCASE2023T2ToyCar). `DCASE202XT2` builds train/valid/test paths, creates `DCASE202XT2Loader` for train data, splits into train/valid, creates one test loader per section. Also defines `width`, `height`, `input_dim` (frames × n_mels × channel). |
| **datasets/dcase_dcase202x_t2_loader.py** | PyTorch `Dataset`: knows paths (dev/eval, machine type, section), calls `loader_common.file_list_generator` and `file_list_to_data`, caches precomputed features in pickle. `__getitem__` returns (data vector, label, condition, basename, index). |
| **datasets/loader_common.py** | **Feature extraction hub:** `file_load` (load WAV), `file_to_vectors` (WAV → log-mel → framed vectors). Also: `file_list_generator` (list normal/anomaly WAVs, labels), `get_machine_type_dict` (read YAML for machine types/sections), download helpers. |

## 3.3 Model and Loss

| File | Role |
|------|------|
| **networks/models.py** | Maps model name string (e.g. "DCASE2023T2-AE") to the class `DCASE2023T2AE`. |
| **networks/base_model.py** | Base class: sets up device, dataset, train/valid/test loaders, paths (logs, checkpoints, result dirs), **fit_anomaly_score_distribution** (gamma fit to scores), **calc_decision_threshold** (quantile of gamma). |
| **networks/dcase2023t2_ae/network.py** | **AENet:** encoder (640→128→…→8), decoder (8→…→640), plus buffers `cov_source`, `cov_target` for Mahalanobis. Forward returns (reconstruction, latent). |
| **networks/dcase2023t2_ae/dcase2023t2_ae.py** | **Training and testing logic:** train loop (MSE or covariance accumulation), validation, covariance pass for Mahalanobis, fitting score distribution, test loop, **anomaly score calculation** (MSE or Mahalanobis), decision (score vs threshold), AUC/pAUC/precision/recall/F1, CSV and plot hooks. |
| **networks/criterion/mahala.py** | **Mahalanobis:** `cov_v_diff`, `cov_v` (covariance of diff), `mahalanobis(u, v, cov, ...)`, `loss_function_mahala(...)`, `calc_inv_cov(model, device)`. |

## 3.4 Evaluation and Plotting

| File | Role |
|------|------|
| **tools/plot_anm_score.py** | Builds boxplot data (normal vs anomaly scores) and saves figure (e.g. anomaly score distribution per section). |
| **tools/plot_loss_curve.py** | Reads training log CSV, plots loss curves (train/val, recon, etc.) and saves PNG. |
| **tools/plot_common.py** | Shared plotting: `Figdata`, `show_figs` (line plots, boxplots, etc.) and save. |
| **tools/extract_results.py** | Reads result CSVs and extracts/summarizes AUC, pAUC, etc. for reporting. |
| **tools/export_results.py** | Aggregates ROC/result CSVs across sections/machine types. |
| **tools/concat_divided_roc.py** | Concatenates per-section ROC result files. |

## 3.5 Scripts

| File | Role |
|------|------|
| **train_ae.sh** | Example script: runs `train.py` with dataset, dev/eval, machine IDs, **--train_only**. |
| **test_ae.sh** | Example script: runs `train.py` with **--test_only** and **--score** (MSE or MAHALA). |

---

# Part 4: Training Process

## 4.1 How the Model Learns

- **Goal:** The AutoEncoder is trained to **reconstruct** its input. So it learns: “normal sound → this compact 8-D representation → back to the same sound.”
- **Data:** Only **normal** (healthy) samples are used for training. The model never sees anomaly labels during training.
- **Idea:** If the model is good at reconstructing normal sounds, it will be **bad** at reconstructing anomalous sounds (higher error). So **reconstruction error** = anomaly score.

**For your RC car:** Train on WAVs of the car running normally. Later, when you feed a recording with a fault, the reconstruction error should be higher.

## 4.2 Loss Function

**Default (MSE):**  
In `dcase2023t2_ae.py`:

```python
loss = F.mse_loss(recon_x, x, reduction="none")  # per element
# then reduced over dimensions and batch
```

So the model minimizes **mean squared error** between input and reconstruction.

**Mahalanobis (optional):**  
During the **covariance pass** (after the last real epoch), the code does **not** train the network. It only:
- Computes `diff = x - recon_x`
- Accumulates covariance of `diff` for “source” and “target” (if present)
- Saves these in `model.cov_source` and `model.cov_target`

So the **training loss** that updates weights is still **MSE**. Mahalanobis is used only for **scoring** after training.

## 4.3 Training Loop (What Happens Each Epoch)

**File:** `networks/dcase2023t2_ae/dcase2023t2_ae.py`, method `train(epoch)`.

1. **Normal epochs (1 to epochs):**
   - Set `model.train()`.
   - For each batch from `train_loader`:
     - Move data to device.
     - Forward: `recon_batch, z = model(data)`.
     - Loss = MSE(recon_batch, data), reduced (e.g. mean over batch).
     - Backward: `loss.backward()`.
     - Optimizer step: `optimizer.step()`.
   - Validation: run over `valid_loader`, compute validation loss (no backward).
   - Collect all training + validation reconstruction errors (scores).
   - **Fit gamma distribution** to these scores and save to pickle (for MSE threshold).
   - Save model checkpoint and append to log CSV.
   - Optionally plot loss curve from log.

2. **Extra “covariance” epoch (epoch = epochs + 1):**
   - No gradient; model in eval mode.
   - For each batch: compute `diff = x - recon_x`, split by source/target, **accumulate covariance** of `diff`.
   - Average covariance, store in `model.cov_source` and `model.cov_target`.
   - Compute inverse covariances.
   - Recompute Mahalanobis scores on train + validation, fit gamma to these scores, save to **Mahalanobis** score-distribution pickle.

So: **training loop** = many epochs of MSE minimization on the AutoEncoder; then one extra pass to build Mahalanobis covariance and score distribution if you use MAHALA.

---

# Part 5: Inference Process (Anomaly Score and Decision)

## 5.1 How Anomaly Scores Are Calculated

**Where:** `networks/dcase2023t2_ae/dcase2023t2_ae.py` in:
- **Training/validation (score collection):** inside `train()` when fitting the score distribution (MSE or Mahalanobis).
- **Test:** inside `eval()` (called from `test()`).

**MSE score:**
- For each batch: `recon_data, _ = model(data)` then `loss = F.mse_loss(recon_data, data, reduction="none")` then mean over the vector and optionally over batch.
- Per file: when the test loader gives one file per batch, the mean MSE over that batch is the **anomaly score** for that file.

**Mahalanobis score:**
- Uses **inverse covariance** computed from the covariance pass (`calc_inv_cov`).
- For each batch: `diff = x - recon_x`; Mahalanobis distance of `diff` with respect to `inv_cov_source` and `inv_cov_target`; score = **min**(distance_source, distance_target) so the “closer” domain is chosen.
- Implemented in `loss_function_mahala` / `mahalanobis` in `networks/criterion/mahala.py`; called from `dcase2023t2_ae.py` in `calc_valid_mahala_score` and in `eval()` when `args.score == "MAHALA"`.

So: **anomaly score is calculated in** `networks/dcase2023t2_ae/dcase2023t2_ae.py` (using `loss_fn` for MSE and `loss_function_mahala` from `networks/criterion/mahala.py` for Mahalanobis).

## 5.2 How the System Decides Normal vs Anomalous

1. **Threshold:**  
   Before testing, the code loads the **fitted gamma** (from the pickle saved during training) and computes the threshold = gamma quantile at `args.decision_threshold` (e.g. 0.9). So 90% of “normal” training scores are below this value.

2. **Decision:**  
   In `eval()`:
   - For each file, one anomaly score is appended to `y_pred`.
   - If `y_pred[-1] > decision_threshold` → decision = 1 (anomaly), else 0 (normal).
   - These are written to the decision-result CSV.

So: **decision** = compare **anomaly score** to **threshold**; threshold comes from **gamma fit to normal scores** in `base_model.calc_decision_threshold`.

---

# Part 6: Evaluation Stage (Metrics and Plots)

## 6.1 What Metrics Are Calculated

**File:** `networks/dcase2023t2_ae/dcase2023t2_ae.py` inside `test()`, in development mode only (when you have labels).

- **AUC (ROC-AUC):** Area Under the ROC Curve. Measures how well scores separate normal (0) vs anomaly (1). Uses `sklearn.metrics.roc_auc_score(y_true, y_pred)`. Can be reported per “source” and “target” domain.
- **pAUC (partial AUC):** Same but only up to a fixed false positive rate (`max_fpr`, e.g. 0.1). Uses `roc_auc_score(..., max_fpr=self.args.max_fpr)`.
- **Precision:** Of the ones predicted as anomaly, how many are truly anomaly. `tp / (tp + fp)`.
- **Recall:** Of the true anomalies, how many were detected. `tp / (tp + fn)`.
- **F1 score:** Balance of precision and recall. `2 * precision * recall / (precision + recall)`.

These can be computed per section and per domain (source/target). Arithmetic and harmonic means over sections are written to the result CSV.

## 6.2 How Results Are Plotted and Reported

- **Anomaly score distribution:**  
  `AnmScoreFigData` (from `tools/plot_anm_score.py`) builds boxplots of scores for normal vs anomaly (per section). Figure is saved in the result directory (e.g. `results/.../..._anm_score.png`).

- **Loss curves:**  
  During training, `csv_to_figdata` (from `tools/plot_loss_curve.py`) reads the log CSV and plots loss, val_loss, recon_loss, etc. Saved next to the log (e.g. in `logs/.../log.png`).

- **Result CSV:**  
  Per-section metrics (AUC, pAUC, precision, recall, F1) and means are written to `result_*_roc.csv` in the result directory.

- **Export/extract:**  
  `tools/export_results.py` and `tools/extract_results.py` aggregate and summarize these CSVs for reporting (e.g. across machine types).

---

# Part 7: Where to Modify for Your Thesis (Mahalanobis and RC Car)

## 7.1 Where Feature Extraction Happens

- **Single place:** `datasets/loader_common.py`, function **`file_to_vectors()`** (and `file_list_to_data()` which loops over it).  
- To change features (e.g. different mel settings, or extra features for RC car), change this file. The rest of the pipeline expects vectors of size `input_dim` (e.g. 640).

## 7.2 Where the Anomaly Score Is Calculated

- **MSE:** `networks/dcase2023t2_ae/dcase2023t2_ae.py`: method `loss_fn(recon_x, x)` and its use in `train()` and `eval()`.
- **Mahalanobis:** Same file uses `loss_function_mahala` and `calc_valid_mahala_score`; the math is in **`networks/criterion/mahala.py`** (`mahalanobis`, `loss_function_mahala`, `calc_inv_cov`).

So: **anomaly score** is computed in **`dcase2023t2_ae.py`** (and the actual distance in **`mahala.py`** for Mahalanobis).

## 7.3 Where to Insert a Mahalanobis-Distance-Based Detector

The repo **already includes** a Mahalanobis option:

- **Covariance:** Computed in `dcase2023t2_ae.py` in the “epochs+1” pass and stored in `model.cov_source` / `model.cov_target`.
- **Score:** In `eval()` and `calc_valid_mahala_score`, when `args.score == "MAHALA"`, the code calls `loss_function_mahala(..., cov=inv_cov_..., use_precision=True)`.

To implement **your** variant (e.g. **Selective Mahalanobis**):

1. **Keep:** Feature extraction (`file_to_vectors`), AutoEncoder training (MSE), and the overall flow in `dcase2023t2_ae.py`.
2. **Change or add:**
   - **`networks/criterion/mahala.py`:** Implement your selective Mahalanobis (e.g. which dimensions or blocks to use, how to compute/regularize covariance). You might add a new function e.g. `loss_function_selective_mahala(...)` and keep the existing `loss_function_mahala` for the baseline.
   - **`networks/dcase2023t2_ae/dcase2023t2_ae.py`:**  
     - In the **covariance pass:** call your new function (or same function with different params) to compute and store the covariance/precision you need.  
     - In **eval()** and **calc_valid_mahala_score:** use your new scoring function instead of (or in addition to) `loss_function_mahala`.  
   - Optionally add a new **`--score`** choice (e.g. `"SELECTIVE_MAHALA"`) in `common.py` and branch in `dcase2023t2_ae.py` on that.

So: **insert your Mahalanobis detector** in the **scoring path**: implement the distance in **`mahala.py`**, then plug it into the **covariance pass and eval** in **`dcase2023t2_ae.py`**. Feature extraction and the AE itself can stay as they are.

---

# Part 8: Simple Glossary

- **AutoEncoder:** Network that compresses input to a small “bottleneck” then reconstructs it. Used to learn “normal” by minimizing reconstruction error.
- **Reconstruction error:** Difference between input and model output (e.g. MSE). Used as anomaly score.
- **Anomaly score:** One number per file; high = more likely anomaly.
- **Decision threshold:** Value from the distribution of normal scores (e.g. 90th percentile). Score > threshold → anomaly.
- **AUC:** Area under ROC curve; how well scores separate normal vs anomaly (0.5 = random, 1.0 = perfect).
- **pAUC:** Partial AUC over a limited false positive range (e.g. 0–10% FPR).
- **Source/target:** In DCASE, different recording conditions or device IDs; the baseline can compute separate covariances and use the minimum distance for scoring.

---

# Part 9: Quick Reference Table

| What you need | Where |
|---------------|--------|
| Load WAV | `datasets/loader_common.py` → `file_load()` |
| WAV → features | `datasets/loader_common.py` → `file_to_vectors()` |
| Dataset / DataLoader | `datasets/datasets.py`, `datasets/dcase_dcase202x_t2_loader.py` |
| Model (AE) | `networks/dcase2023t2_ae/network.py` → `AENet` |
| Training loop & loss | `networks/dcase2023t2_ae/dcase2023t2_ae.py` → `train()` |
| MSE loss | `dcase2023t2_ae.py` → `loss_fn()` |
| Mahalanobis math | `networks/criterion/mahala.py` |
| Covariance pass | `dcase2023t2_ae.py` → `train(epoch)` when `epoch == epochs+1` |
| Anomaly score (MSE or Mahala) | `dcase2023t2_ae.py` → `eval()`, `calc_valid_mahala_score()` |
| Threshold | `networks/base_model.py` → `fit_anomaly_score_distribution()`, `calc_decision_threshold()` |
| Decision (0/1) | `dcase2023t2_ae.py` → `eval()` (compare score to threshold) |
| AUC, pAUC, precision, recall, F1 | `dcase2023t2_ae.py` → `test()` (dev mode) |
| Score boxplots | `tools/plot_anm_score.py` |
| Loss curves | `tools/plot_loss_curve.py` |
| Add Selective Mahalanobis | `mahala.py` (new distance) + `dcase2023t2_ae.py` (covariance + scoring branch) |

This should give you a clear, structured understanding of the whole repository and where to plug in your toy RC car data and Mahalanobis-based detector.
