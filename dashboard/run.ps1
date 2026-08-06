# run.ps1 — start the dashboard.   .\run.ps1
# Stop it with Ctrl+C. Run .\setup.ps1 first on a new machine.

$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $venvPy = "py"          # fall back to the system Python
}

# --theme.base=light keeps the toolbar and sidebar readable even if
# .streamlit\config.toml is missing or unreadable.
& $venvPy -m streamlit run app.py --theme.base=light
