$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error "Не знайдено .venv: $python"
    exit 1
}

$argsList = @(
    'run_pipeline.py',
    '--input', 'training_2025-01-01_2025-02-01.parquet',
    '--threshold', '1',
    '--interval', '300',
    '--blind-spots-output', 'blind_spots_for_simulation.csv',
    '--simulation-output', 'simulation_points.csv',
    '--fused-output', 'fused_dataset.parquet'
)

& $python @argsList
exit $LASTEXITCODE
