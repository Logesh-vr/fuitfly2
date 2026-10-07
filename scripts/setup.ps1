$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.python'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ then rerun this script.'
}
uv python install 3.12.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    uv venv --python 3.12.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Output 'Ready: .\.venv\Scripts\python.exe -m virtual_fly body'
