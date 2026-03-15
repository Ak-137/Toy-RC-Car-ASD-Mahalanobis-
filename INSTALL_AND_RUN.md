# Install and Run Instructions (Windows)

## What was fixed

The original `requirements.txt` used `torch==2.6.0+cu118`, which **is not on the default PyPI**. Pip could not find that package, so the install failed.

**Fix:** PyTorch was removed from `requirements.txt` and must be installed separately (see Step 2 below). All other dependencies are in `requirements.txt` and install with pip.

---

## Step 1: Install dependencies from requirements.txt

Open **Command Prompt** or **PowerShell** and run:

```cmd
cd c:\Users\amrkh\OneDrive\Desktop\ToyRCCar_ASD\Toy-RC-Car-ASD-Mahalanobis-
pip install -r requirements.txt
```

If the folder name on your PC is different (e.g. no trailing hyphen), use the correct path. This installs scipy, librosa, matplotlib, tqdm, seaborn, fasteners, PyYAML, numpy, pandas, scikit-learn and their dependencies.

---

## Step 2: Install PyTorch

**Option A – CPU only (works on any PC, recommended for a first test):**

```cmd
pip install torch torchvision
```

**Option B – GPU with CUDA 11.8 (only if you have an NVIDIA GPU and CUDA 11.8):**

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: Download the dataset (required to run training/test)

The code expects the DCASE 2023 Task 2 development data. For a **quick test with ToyCar**:

1. Create the data folder:

   ```cmd
   mkdir "c:\Users\amrkh\OneDrive\Desktop\ToyRCCar_ASD\Toy-RC-Car-ASD-Mahalanobis-\data\dcase2023t2\dev_data\raw"
   ```

2. Download the ToyCar dev zip (about 500 MB) from:  
   https://zenodo.org/record/7882613/files/dev_ToyCar.zip  

   Save it to:  
   `...\data\dcase2023t2\dev_data\raw\`  
   and unzip it there.

3. **Important:** The zip creates `dev_ToyCar\ToyCar\train` and `...\test`. The code expects `ToyCar` directly under `raw`. So after unzipping, **move** the inner `ToyCar` folder:
   - **From:** `data\dcase2023t2\dev_data\raw\dev_ToyCar\ToyCar`  
   - **To:**   `data\dcase2023t2\dev_data\raw\ToyCar`  
   So that you have `raw\ToyCar\train` and `raw\ToyCar\test` with WAV files inside.

---

## Step 4: Run training and test

From the project folder, run:

```cmd
cd c:\Users\amrkh\OneDrive\Desktop\ToyRCCar_ASD\Toy-RC-Car-ASD-Mahalanobis-
python train.py --dataset DCASE2023T2ToyCar --dev --epochs 2
```

- **`--dev`** uses the development dataset (required).
- **`--epochs 2`** keeps the first run short (about a few minutes). For real training, use more epochs (e.g. 100 as in `baseline.yaml`).

This will:

1. Train the AutoEncoder on normal ToyCar sounds for 2 epochs.  
2. Run the test/evaluation and print AUC, pAUC, etc.  
3. Write results under `results/` and save the model under `models/`.

---

## Step 5: Run test only (after training once)

If you have already trained and only want to run evaluation:

```cmd
cd c:\Users\amrkh\OneDrive\Desktop\ToyRCCar_ASD\Toy-RC-Car-ASD-Mahalanobis-
python train.py --dataset DCASE2023T2ToyCar --dev --score MSE --test_only
```

Use `--score MAHALA` if you trained with Mahalanobis and want to evaluate with that score.

---

## Troubleshooting

- **“FileNotFoundError” or “is not directory” for data:**  
  Complete Step 3: create the folder, download `dev_ToyCar.zip`, and unzip so that `data/dcase2023t2/dev_data/raw/ToyCar/train` and `.../test` exist with WAV files.

- **“No module named 'torch'”:**  
  Run Step 2 (install PyTorch).

- **“No module named 'librosa'” (or scipy, etc.):**  
  Run Step 1 again from the correct project folder.

- **CUDA/GPU errors:**  
  Use the CPU install (Step 2, Option A). The code will run on CPU; it only uses GPU if available.
