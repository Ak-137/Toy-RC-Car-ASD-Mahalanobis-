@echo off
REM Quick test: 2 epochs train + evaluation. Requires ToyCar dev data (see INSTALL_AND_RUN.md).
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

python train.py --dataset DCASE2023T2ToyCar --dev --epochs 2
if errorlevel 1 (
    echo Run failed. Make sure you installed dependencies and downloaded the ToyCar dev data.
    echo See INSTALL_AND_RUN.md for details.
)
pause
