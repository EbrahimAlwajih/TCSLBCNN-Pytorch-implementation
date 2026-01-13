param(
  [ValidateSet("lint","test","smoke")] [string]$Task="test"
)

$Py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

if (!(Test-Path $Py)) {
  Write-Error "Virtualenv not found at .venv. Create it first: python -m venv .venv"
  exit 1
}

if ($Task -eq "lint") {
  & $Py -m ruff check .
  exit $LASTEXITCODE
}

if ($Task -eq "test") {
  & $Py -m pytest -q
  exit $LASTEXITCODE
}

if ($Task -eq "smoke") {
  & $Py -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
  exit $LASTEXITCODE
}
