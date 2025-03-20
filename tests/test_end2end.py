import lightning

def test_model_trainer_fit(multimodal_model, sample_train_val_datamodule):
    batch = next(iter(sample_train_val_datamodule.train_dataloader()))

    print("Keys in batch:", batch.keys())
    if "solar_azimuth" in batch:
        print("solar_azimuth shape:", batch["solar_azimuth"].shape)
    if "solar_elevation" in batch:
        print("solar_elevation shape:", batch["solar_elevation"].shape)

    y = multimodal_model(batch)
    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model=multimodal_model, datamodule=sample_train_val_datamodule)
