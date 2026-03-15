# DCASE Task 2 Baseline: A Beginner-Friendly Guide

This guide explains how the anomalous sound detection (ASD) baseline works: **data flow**, **audio processing**, **anomaly scoring**, and **training/evaluation**. It is written for someone with programming experience but limited ML/audio background, and relates concepts to your toy RC car application.

---

## 1. High-Level Idea (Same for Toy RC Car)

**Goal:** Learn what “normal” sounds like, then flag anything that doesn’t match.

- **Training:** Use only **normal** (healthy) audio. The system learns a compact representation of “normal.”
- **Testing:** For each new audio clip, compute an **anomaly score**. High score → likely anomaly (e.g. wear, fault); low score → likely normal.

For a toy RC car: normal = healthy motor/gearbox; anomaly = unusual rattling, grinding, or whine from early wear or failure.

---

## 2. Data Flow (Step-by-Step)

End-to-end path from audio files to final decision:

```
WAV files  →  Log-mel features  →  AutoEncoder  →  Reconstruction error  →  Anomaly score  →  Decision (normal/anomaly)
```

### 2.1 Where Things Live in Code

| Step | What happens | Where in code |
|------|----------------|---------------|
| 1. Config | Load dataset name, model, feature params, paths | `train.py` → `common.yaml_load()` → `baseline.yaml` |
| 2. Dataset | Build train/valid/test from WAVs and convert to features | `datasets/datasets.py` (DCASE202XT2) → `dcase_dcase202x_t2_loader.py` → `loader_common.file_to_vectors()` |
| 3. Model | AutoEncoder: encode → bottleneck → decode | `networks/dcase2023t2_ae/network.py` (AENet) |
| 4. Training | Minimize reconstruction error (MSE or Mahalanobis); optional: fit covariance | `networks/dcase2023t2_ae/dcase2023t2_ae.py` (train) |
| 5. Score & threshold | Score = reconstruction error; threshold from score distribution | `dcase2023t2_ae.py` (eval, calc_decision_threshold); `base_model.py` (fit_anomaly_score_distribution, calc_decision_threshold) |
| 6. Decision | score > threshold → anomaly (1), else normal (0) | `dcase2023t2_ae.py` (eval: decision_result_list) |

### 2.2 Flow for One Audio File at Test Time

1. **Load WAV** → raw waveform (e.g. 10 s at 16 kHz).
2. **Feature extraction** → many **feature vectors** per file (one per “frame”); each vector has size `frames × n_mels` (e.g. 5×128 = 640).
3. **AutoEncoder** → each vector is encoded to a small latent (e.g. 8-D), then decoded back to 640-D.
4. **Reconstruction error** → compare decoded vs input (MSE or Mahalanobis). Per-file score is typically the **mean** (or min over source/target in Mahalanobis mode) of these errors.
5. **Decision** → compare score to a **decision threshold**; if score > threshold → anomaly.

So: **one WAV → many vectors → many errors → one score per file → one binary decision per file.**

---

## 3. How Audio Is Processed

Audio is not fed as raw waveform to the model. It is converted to **log-mel spectrogram** and then to **fixed-size vectors**.

### 3.1 WAV → Mel spectrogram

- **WAV:** sequence of samples (amplitude over time).
- **STFT (Short-Time Fourier Transform):** split signal into short windows, compute frequency content per window. Gives a **spectrogram** (time × frequency).
- **Mel filterbank:** human (and many machine) sounds are better described on a **mel** scale (low frequencies more resolved). The code uses `n_mels` (e.g. 128) mel bands.
- **Mel spectrogram:** magnitude (or power) in each mel band per time frame.
- **Log:** take log (e.g. log10). This compresses dynamic range and matches how we perceive loudness.

Relevant parameters in `common.py` / `baseline.yaml`:

- `n_fft`: FFT length (e.g. 1024).
- `hop_length`: samples between consecutive frames (e.g. 512).
- `n_mels`: number of mel bands (e.g. 128).
- `power`: exponent for magnitude (e.g. 2.0 for power).

So: **WAV → STFT → mel bands → power → log** → 2D array of shape `(n_mels, time_frames)`.

### 3.2 Mel spectrogram → Feature vectors (framing)

The model expects **fixed-length** inputs. So the 2D log-mel spectrogram is turned into many vectors:

- **Frames:** use a sliding window of `frames` consecutive time steps (e.g. 5).
- Each window is flattened: `frames × n_mels` = one vector (e.g. 5×128 = 640).
- **Hop:** `frame_hop_length` (e.g. 1) controls how much the window moves in time.

So one WAV produces many overlapping vectors; each vector is one “snippet” of the sound in time-frequency.

**Code:** `loader_common.file_to_vectors()` builds these vectors; `dcase_dcase202x_t2_loader.file_list_to_data()` does it for a list of files and optionally subsamples with `n_hop_frames`.

### 3.3 Dimensions Used by the Model

From `datasets/datasets.py` (DCASE202XT2):

- `width = args.frames` (e.g. 5)
- `height = args.n_mels` (e.g. 128)
- `channel = 1`
- `input_dim = width × height × channel` (e.g. 640)

So the **input** to the AutoEncoder is a batch of vectors of size `input_dim`. For your toy RC car you would use the same pipeline: same framing and mel settings (or tune them later); only the WAV content (and possibly dataset name) changes.

---

## 4. How Anomaly Scores Are Calculated

The baseline supports two scoring modes: **MSE** and **MAHALA** (Mahalanobis). Both start from **reconstruction error**.

### 4.1 AutoEncoder output

- **Input:** feature vector `x` (e.g. 640-D).
- **Encoder:** several linear + BatchNorm + ReLU layers, down to a **bottleneck** (e.g. 8-D) `z`.
- **Decoder:** mirror back to 640-D.
- **Output:** reconstructed vector `recon_x`.

So the model learns to compress “normal” sounds into a small space and reconstruct them. Unusual sounds (anomalies) are expected to reconstruct poorly.

### 4.2 MSE score (default)

- **Error:** `MSE(recon_x, x)` per element, then typically averaged over the vector (and then over frames for a file).
- **Interpretation:** large average squared difference → input was hard to reconstruct → likely anomaly.

Code: `dcase2023t2_ae.py` uses `F.mse_loss(recon_x, x, reduction="none")` then reduction (e.g. mean over dim 1, then over samples).

### 4.3 Mahalanobis score (optional)

- **Idea:** Not all dimensions of the error are equally variable. Mahalanobis distance uses the **covariance** of the reconstruction error (on normal data) to weight dimensions: more variable directions count less.
- **Training (extra step):** After the last epoch, the code runs once more over the training (and validation) data **without** updating weights. It computes:
  - `diff = x - recon_x`
  - Covariance of `diff` for “source” and “target” domains (if present), stored in `model.cov_source` and `model.cov_target`.
- **Scoring:** For each test sample, compute Mahalanobis distance of `diff` using the inverse covariance (precision matrix). The **anomaly score** is the **minimum** of the distance under “source” and “target” covariance (so the system picks the more likely domain).

Code: `networks/criterion/mahala.py` (`loss_function_mahala`, `mahalanobis`, `calc_inv_cov`); in `dcase2023t2_ae.py`, covariance is accumulated in the “epochs+1” pass, then inverse covariance is computed and used in `calc_valid_mahala_score` and in `eval()` when `args.score == "MAHALA"`.

### 4.4 Per-file score

- During **evaluation**, the loader often gives one batch per file (batch size = number of vectors in that file). For each batch you get one scalar score (e.g. mean MSE or min over source/target Mahalanobis).
- That scalar is the **anomaly score** for that file; it is written to the anomaly score CSV and then compared to the decision threshold.

---

## 5. Decision Threshold (Normal vs Anomaly)

Scores are continuous. To get a binary decision (normal/anomaly), the baseline fits a **distribution** to normal (training/validation) scores and sets a **threshold**.

### 5.1 Fitting the score distribution

- After each training epoch (and after the covariance pass for Mahalanobis), the code collects all reconstruction (or Mahalanobis) scores on **training + validation** data.
- It fits a **gamma distribution** to these scores (shape, location, scale).
- These parameters are saved to a pickle file (e.g. `score_distr_..._mse.pickle` or `..._mahala.pickle`).

Code: `base_model.fit_anomaly_score_distribution()` uses `scipy.stats.gamma.fit(y_pred)`.

### 5.2 Choosing the threshold

- A single number `decision_threshold` (e.g. 0.9) is used as a **quantile** of the fitted gamma.
- Threshold = value such that 90% of the fitted (normal) distribution is below it; i.e. 10% of “normal” would be above it (false positives).
- At test time: **if anomaly_score > threshold → anomaly (1), else normal (0).**

Code: `base_model.calc_decision_threshold()` loads the pickle and uses `scipy.stats.gamma.ppf(decision_threshold, shape, loc, scale)`.

So you don’t manually set a score value; you set a **quantile** (e.g. 0.9) and the threshold is derived from the normal score distribution.

---

## 6. How Training Works

### 6.1 What is trained

- **Parameters:** only the AutoEncoder (encoder + decoder). The Mahalanobis covariance matrices are **not** trained by gradient descent; they are computed in a separate pass and stored as non-trainable parameters.

### 6.2 One epoch (normal training)

1. For each batch of feature vectors from the **training** set:
   - Forward: `recon_batch, z = model(data)`.
   - Loss = reconstruction error (MSE or, when updating covariance, a placeholder MSE-like term).
   - Backward + optimizer step.
2. Optionally, same for **validation** set (no backward) to compute validation loss and to collect scores for fitting the score distribution.
3. Fit gamma to collected scores and save distribution parameters.
4. Save model checkpoint and logs.

### 6.3 Extra “covariance” epoch (when using Mahalanobis)

- After the last real epoch, the code runs **one more** pass over train (and then validation) data with **no gradient**.
- For each batch it computes `diff = x - recon_x`, splits by source/target, and **accumulates** covariance of `diff`.
- After the pass, it averages (e.g. divide by count-1), stores in `model.cov_source` and `model.cov_target`, then computes inverse covariances.
- Then it recomputes Mahalanobis scores on train+validation, fits the gamma to these scores, and saves the Mahalanobis score distribution pickle.

So: **training = AutoEncoder only;** Mahalanobis is a **post-hoc** step that uses the fixed AutoEncoder and the train/validation data to define the score and threshold.

---

## 7. How Evaluation Works

### 7.1 Development vs evaluation mode

- **Development:** you have labels (normal/anomaly). The code can compute AUC, pAUC, precision, recall, F1, and save anomaly scores and decision results.
- **Evaluation:** no labels; the code only writes anomaly scores and decisions (for submission or later analysis).

Mode is determined by `--dev` or `--eval` and by the dataset (dev_data vs eval_data).

### 7.2 Test loop (per section / machine ID)

- For each test section (machine type + section ID), a **test loader** is created that loads all test files for that section.
- For each file (or batch of vectors from one file):
  - Forward through the model.
  - Compute anomaly score (MSE or Mahalanobis, as chosen by `--score`).
  - Append to list of scores and decisions (1 if score > threshold, 0 else).
- Results are written to CSV: anomaly scores and decision results.

### 7.3 Metrics (development only)

- **AUC:** area under ROC curve (higher = better separation of normal vs anomaly).
- **pAUC:** partial AUC up to a fixed false positive rate (`max_fpr`, e.g. 0.1).
- **Precision, recall, F1:** from the binary decisions at the chosen threshold.

The baseline can report these per “source” and “target” domain when both exist (e.g. different recording conditions or device IDs). For a single toy RC car, you might only have one domain.

---

## 8. Summary: From Your Perspective (Toy RC Car)

- **Data:** You will have WAVs of your RC car (normal = healthy, anomaly = with induced or natural faults). Place them in the same folder/label structure as DCASE (or adapt the loader).
- **Features:** Same pipeline: WAV → log-mel spectrogram → framed vectors (e.g. 5×128). No change needed to start.
- **Model:** Same AutoEncoder: learns to reconstruct “normal” sound; high reconstruction error (or high Mahalanobis distance) → anomaly.
- **Scores:** Start with MSE; later you can switch to Mahalanobis (`--score MAHALA`) for your thesis (Selective Mahalanobis, etc.).
- **Threshold:** Fitted from normal scores (gamma); you only choose the quantile (e.g. 0.9).
- **Training:** Train on normal-only data; optionally use Mahalanobis covariance pass. **Evaluation:** run on test WAVs, get scores and decisions.

Once you are comfortable with this baseline (data flow, audio processing, anomaly score, training, and evaluation), you can then change the **scoring** part (e.g. Selective Mahalanobis) while keeping the rest of the pipeline the same.

---

## 9. Quick Reference: Important Files

| Purpose | File |
|--------|------|
| Entry point | `train.py` |
| Config | `baseline.yaml`, `common.py` |
| Dataset & loaders | `datasets/datasets.py`, `datasets/dcase_dcase202x_t2_loader.py` |
| Audio → features | `datasets/loader_common.py` (`file_to_vectors`, `file_list_to_data`) |
| Model | `networks/dcase2023t2_ae/network.py` (AENet), `dcase2023t2_ae.py` (training & eval) |
| Loss / Mahalanobis | `networks/criterion/mahala.py` |
| Score distribution & threshold | `networks/base_model.py` |

If you want, the next step can be a short “runbook” (exact commands and `baseline.yaml` changes) to train and test on one DCASE machine type (e.g. ToyCar) so you can see the flow end-to-end on your machine.
