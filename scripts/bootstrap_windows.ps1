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

function Set-NumericThreadDefaults {
    $threadVars = @(
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS"
    )
    foreach ($name in $threadVars) {
        if (-not (Get-Item "Env:$name" -ErrorAction SilentlyContinue)) {
            Set-Item -Path "Env:$name" -Value "1"
        }
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

function Get-CognitiveDatasetPath($RootPath) {
    return (Join-Path $RootPath "data\Cognitive Radio Spectrum Sensing Dataset.csv")
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
Step "Configuring numeric runtime defaults for Windows..."
Set-NumericThreadDefaults
Step "Syncing dependencies..."
uv sync --dev

if ($DatasetSource -eq "auto" -or $DatasetSource -eq "kaggle" -or $DatasetSource -eq "cognitive") {
    $localDataset = Get-CognitiveDatasetPath -RootPath $root
    if (Test-Path $localDataset) {
        Step "Using existing local Cognitive Radio dataset: $localDataset"
    } else {
        Ensure-Kaggle
        Download-CognitiveDataset -RootPath $root -Slug $KaggleDataset
    }
}

if ($IncludeVSB) {
    Ensure-Kaggle
    Download-VSB -RootPath $root -CompetitionId $VsbCompetition
}

if ($RunPipeline) {
    Step "Running quickstart pipeline..."
    uv run python -m emc_diag quickstart --device auto --output-root artifacts
}

Step "Bootstrap completed."
