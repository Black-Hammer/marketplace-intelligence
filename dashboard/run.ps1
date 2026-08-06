# setup.ps1 — one-time setup for the Marketplace Intelligence dashboard.
# Run from the folder holding app.py:   .\setup.ps1
# If PowerShell refuses to run it:      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Marketplace Intelligence — setup" -ForegroundColor Cyan
Write-Host "--------------------------------"

# 1. Are we in the right folder?
if (-not (Test-Path ".\app.py")) {
    Write-Host "app.py is not in this folder." -ForegroundColor Red
    Write-Host "Open the dashboard folder first, e.g.  cd `"$HOME\marketplace-dashboard`"" 
    exit 1
}
if ($PWD.Path -like "*Program Files*") {
    Write-Host "This folder is under Program Files, which Windows protects." -ForegroundColor Red
    Write-Host "Move the files to $HOME\marketplace-dashboard and run this again."
    exit 1
}

# 2. Find a working Python. The bare 'python' command is often the Store stub.
$py = $null
foreach ($candidate in @("py", "python3", "python")) {
    try {
        $version = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python 3") { $py = $candidate; break }
    } catch { }
}
if (-not $py) {
    Write-Host "No working Python found." -ForegroundColor Red
    Write-Host "Install it, TICKING 'Add python.exe to PATH' on the first screen:"
    Write-Host "    winget install Python.Python.3.12"
    Write-Host "Then CLOSE this window, open a new one, and run .\setup.ps1 again."
    exit 1
}
Write-Host "Python:      $(& $py --version)  (via '$py')" -ForegroundColor Green

# 3. Virtual environment, so upgrades elsewhere can never break this app.
if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating the virtual environment..."
    & $py -m venv .venv
}
$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "The virtual environment did not build; falling back to the system Python." -ForegroundColor Yellow
    $venvPy = $py
}

# 4. Dependencies.
Write-Host "Installing dependencies — this takes a few minutes on a fresh machine..."
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet -r requirements.txt

# 5. Theme config, written without a byte-order mark (Out-File would add one
#    and Streamlit's TOML parser rejects it).
# Written to the repository root when there is one, because that is where both
# Streamlit locally and Community Cloud look for it.
$configRoot = if (Test-Path "..\data\processed") { (Resolve-Path "..").Path } else { $PWD.Path }
if (-not (Test-Path "$configRoot\.streamlit")) {
    New-Item -ItemType Directory -Path "$configRoot\.streamlit" | Out-Null
}
$toml = "[theme]`nbase = `"light`"`nprimaryColor = `"#0E6B60`"`nbackgroundColor = `"#F3F5F4`"`nsecondaryBackgroundColor = `"#FFFFFF`"`ntextColor = `"#12211F`"`n"
[IO.File]::WriteAllText("$configRoot\.streamlit\config.toml", $toml)

Write-Host ""
Write-Host "Ready. Start the dashboard with:  .\run.ps1" -ForegroundColor Green
Write-Host ""
