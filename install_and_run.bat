@echo off
REM Run this from the Toy-RC-Car-ASD-Mahalanobis- folder (or use full path to it).
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo ========== Step 1: Install dependencies from requirements.txt ==========
pip install -r requirements.txt
if errorlevel 1 (
    echo pip install failed. Check your Python/pip and try again.
    pause
    exit /b 1
)

echo.
echo ========== Step 2: Install PyTorch (CPU) ==========
pip install torch torchvision
if errorlevel 1 (
    echo PyTorch install failed.
    pause
    exit /b 1
)

echo.
echo ========== Install done. ==========
echo.
echo To RUN training and test you need the dataset first.
echo 1. Create folder: data\dcase2023t2\dev_data\raw
echo 2. Download dev_ToyCar.zip from https://zenodo.org/record/7882613/files/dev_ToyCar.zip
echo 3. Unzip it so that data\dcase2023t2\dev_data\raw\ToyCar\train and test exist.
echo.
echo Then run:
echo   python train.py --dataset DCASE2023T2ToyCar --dev --epochs 2
echo.
echo Or run "run_test.bat" after you have the data.
pause
