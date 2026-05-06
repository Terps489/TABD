# Скрипт установки conda-окружения tabd_tft
# Запускать один раз: powershell -ExecutionPolicy Bypass -File setup_env.ps1

$condaPath = "C:\Users\Admin\anaconda3"
$envName = "tabd_tft"
$pythonPath = "$condaPath\envs\$envName\python.exe"

Write-Host "=== Установка окружения TABD TFT ===" -ForegroundColor Cyan

# Шаг 1: Создание conda-окружения
Write-Host "`n[1/4] Создание conda-окружения '$envName' с Python 3.11..." -ForegroundColor Yellow
& "$condaPath\Scripts\conda.exe" create -n $envName python=3.11 -y
if (-not $?) { Write-Host "Не удалось создать conda-окружение" -ForegroundColor Red; exit 1 }

# Шаг 2: Установка PyTorch с CUDA 12.4
Write-Host "`n[2/4] Установка PyTorch с CUDA 12.4..." -ForegroundColor Yellow
& $pythonPath -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
& $pythonPath -m pip install sympy==1.13.1 typing_extensions filelock jinja2 networkx fsspec pillow numpy
if (-not $?) { Write-Host "Не удалось установить PyTorch" -ForegroundColor Red; exit 1 }

# Шаг 3: Установка зависимостей проекта
Write-Host "`n[3/4] Установка зависимостей проекта..." -ForegroundColor Yellow
& $pythonPath -m pip install -r requirements.txt
if (-not $?) { Write-Host "Не удалось установить зависимости" -ForegroundColor Red; exit 1 }

# Шаг 4: Проверка
Write-Host "`n[4/4] Проверка установки..." -ForegroundColor Yellow
& $pythonPath -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
& $pythonPath -c "import pytorch_forecasting; print(f'pytorch-forecasting: {pytorch_forecasting.__version__}')"
& $pythonPath -c "import dash; print(f'Dash: {dash.__version__}')"

Write-Host "`n=== Установка завершена! ===" -ForegroundColor Green
Write-Host "Запуск проекта: powershell -ExecutionPolicy Bypass -File run.ps1" -ForegroundColor Cyan
