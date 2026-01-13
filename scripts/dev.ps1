param(
  [ValidateSet("lint","test","smoke")] [string]$Task="test"
)

if ($Task -eq "lint") {
  ruff check .
  exit $LASTEXITCODE
}

if ($Task -eq "test") {
  pytest -q
  exit $LASTEXITCODE
}

if ($Task -eq "smoke") {
  python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
  exit $LASTEXITCODE
}
