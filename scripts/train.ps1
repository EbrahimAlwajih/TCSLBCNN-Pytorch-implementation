param(
  [ValidateSet("cifar10","mnist")] [string]$Dataset="cifar10",
  [int]$Epochs=5,
  [int]$Depth=2,
  [int]$BatchSize=64
)

python -c "from tcslbcnn.training import train; train(dataset='$Dataset', n_epochs=$Epochs, tcslbcnn_depth=$Depth, batch_size=$BatchSize)"
