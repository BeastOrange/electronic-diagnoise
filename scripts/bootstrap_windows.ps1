param(
    [string]$ProjectRoot = ".",
    [string]$DatasetSource = "auto",
    [string]$KaggleDataset = "ajithdari/cognitive-radio-spectrum-sensing-dataset",
    [string]$VsbCompetition = "vsb-power-line-fault-detection",
    [switch]$IncludeVSB,
    [switch]$RunPipeline
)

$ErrorActionPreference = "Stop"

function Step($Text) {
    Write-Host "[bootstrap] $Text" -ForegroundColor Cyan
}

function Ensure-Command($Name, $InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name not found. $InstallHint"
    }
}

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }
    Step "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.local\bin;$env:Path"
    Ensure-Command "uv" "Please restart PowerShell and run the script again."
}

function Ensure-Kaggle {
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        Step "Installing kaggle CLI with uv tool..."
        uv tool install kaggle
    }
    $env:Path = "$HOME\.local\bin;$env:Path"
    Ensure-Command "kaggle" "Configure PATH so kaggle is available."
    if (-not $env:KAGGLE_USERNAME -or -not $env:KAGGLE_KEY) {
        Write-Host "Warning: KAGGLE_USERNAME or KAGGLE_KEY not set. Kaggle downloads may fail." -ForegroundColor Yellow
    }
}

function Resolve-ProjectRoot($InputPath) {
    return (Resolve-Path $InputPath).Path
}

function Download-CognitiveDataset($RootPath, $Slug) {
    $dataDir = Join-Path $RootPath "data"
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    Step "Downloading Cognitive Radio dataset ($Slug)..."
    kaggle datasets download -d $Slug -p $dataDir --force
    Get-ChildItem $dataDir -Filter "*.zip" | ForEach-Object {
        Expand-Archive -Path $_.FullName -DestinationPath $dataDir -Force
    }
}

function Download-VSB($RootPath, $CompetitionId) {
    $dataDir = Join-Path $RootPath "data\vsb-power-line-fault-detection"
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    Step "Downloading VSB competition files ($CompetitionId)..."
    kaggle competitions download -c $CompetitionId -p $dataDir
    Get-ChildItem $dataDir -Filter "*.zip" | ForEach-Object {
        Expand-Archive -Path $_.FullName -DestinationPath $dataDir -Force
    }
}

$root = Resolve-ProjectRoot $ProjectRoot
Set-Location $root

Step "Project root: $root"
Ensure-Uv
Step "Syncing dependencies..."
uv sync --dev

if ($DatasetSource -eq "auto" -or $DatasetSource -eq "kaggle" -or $DatasetSource -eq "cognitive") {
    Ensure-Kaggle
    Download-CognitiveDataset -RootPath $root -Slug $KaggleDataset
}

if ($IncludeVSB) {
    Ensure-Kaggle
    Download-VSB -RootPath $root -CompetitionId $VsbCompetition
}

if ($RunPipeline) {
    Step "Running Cognitive Radio CNN-first pipeline..."
    uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
    uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
    uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device auto
}

Step "Bootstrap completed."
