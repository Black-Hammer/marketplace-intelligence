# run.ps1 — start the dashboard.   .\run.ps1
# Stop it with Ctrl+C. Run .\setup.ps1 first on a new machine.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { $venvPy = "py" }   # fall back to system Python

# Launch from the repository root when there is one, so the sidebar's default
# data/processed path resolves and the app opens on the real result tables.
$root = Split-Path -Parent $here
if (Test-Path (Join-Path $root "data\processed")) {
    Set-Location $root
    $entry = "dashboard\app.py"
} else {
    Set-Location $here
    $entry = "app.py"
}

Write-Host "Launching from $(Get-Location)" -ForegroundColor Cyan
& $venvPy -m streamlit run $entry --theme.base=light
