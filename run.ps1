# Main runner script for TABD TFT project
# Usage: powershell -ExecutionPolicy Bypass -File run.ps1 [-Mode train|predict|dashboard|all] [-Quick]

param(
    [string]$Mode = "all",
    [switch]$Quick
)

$condaPath = "C:\Users\Admin\anaconda3"
$envName = "tabd_tft"
$pythonPath = "$condaPath\envs\$envName\python.exe"
$projectDir = "D:\project\TABD"

# Check environment
if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Conda environment '$envName' not found." -ForegroundColor Red
    Write-Host "Run: powershell -ExecutionPolicy Bypass -File setup_env.ps1" -ForegroundColor Yellow
    exit 1
}

Set-Location $projectDir

$args = @("run.py", "--mode", $Mode)
if ($Quick) { $args += "--quick" }

Write-Host "=== TABD TFT Pipeline ===" -ForegroundColor Cyan
Write-Host "Mode: $Mode  |  Quick: $Quick" -ForegroundColor Yellow
Write-Host ""

& $pythonPath @args
