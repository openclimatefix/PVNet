import lightning
from pvnet.models.lightning_module import PVNetLightningModule
from pvnet.optimizers import EmbAdamWReduceLROnPlateau

def test_model_trainer_fit(multimodal_model, sample_train_val_datamodule):
    """Test end-to-end training."""

    ligtning_model = PVNetLightningModule(
        model=multimodal_model, 
        optimizer=EmbAdamWReduceLROnPlateau(),
    )

    # Get a sample batch for testing
    batch = next(iter(sample_train_val_datamodule.train_dataloader()))

    # Run a forward pass to verify the training module works with the data
    y = multimodal_model(batch)

    # Train the model for one batch
    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model=ligtning_model, datamodule=sample_train_val_datamodule)
