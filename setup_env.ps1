# Setup script for tabd_tft conda environment
# Run once: powershell -ExecutionPolicy Bypass -File setup_env.ps1

$condaPath = "C:\Users\Admin\anaconda3"
$envName = "tabd_tft"
$pythonPath = "$condaPath\envs\$envName\python.exe"

Write-Host "=== TABD TFT Environment Setup ===" -ForegroundColor Cyan

# Step 1: Create conda environment
Write-Host "`n[1/4] Creating conda environment '$envName' with Python 3.11..." -ForegroundColor Yellow
& "$condaPath\Scripts\conda.exe" create -n $envName python=3.11 -y
if (-not $?) { Write-Host "Failed to create conda env" -ForegroundColor Red; exit 1 }

# Step 2: Install PyTorch with CUDA 12.4
Write-Host "`n[2/4] Installing PyTorch with CUDA 12.4..." -ForegroundColor Yellow
& $pythonPath -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
& $pythonPath -m pip install sympy==1.13.1 typing_extensions filelock jinja2 networkx fsspec pillow numpy
if (-not $?) { Write-Host "Failed to install PyTorch" -ForegroundColor Red; exit 1 }

# Step 3: Install project dependencies
Write-Host "`n[3/4] Installing project dependencies..." -ForegroundColor Yellow
& $pythonPath -m pip install -r requirements.txt
if (-not $?) { Write-Host "Failed to install dependencies" -ForegroundColor Red; exit 1 }

# Step 4: Verify
Write-Host "`n[4/4] Verifying installation..." -ForegroundColor Yellow
& $pythonPath -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
& $pythonPath -c "import pytorch_forecasting; print(f'pytorch-forecasting: {pytorch_forecasting.__version__}')"
& $pythonPath -c "import dash; print(f'Dash: {dash.__version__}')"

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "To run the project: powershell -ExecutionPolicy Bypass -File run.ps1" -ForegroundColor Cyan
