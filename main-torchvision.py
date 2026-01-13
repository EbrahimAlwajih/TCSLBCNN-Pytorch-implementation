from tcslbcnn.training import train

if __name__ == "__main__":
    train(
        dataset="cifar10",
        n_epochs=80,
        tcslbcnn_depth=15,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lr_scheduler_step=30,
    )
