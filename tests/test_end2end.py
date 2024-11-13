import lightning


def test_model_trainer_fit(multimodal_model, sample_train_val_datamodule):
    batch = next(iter(sample_train_val_datamodule.train_dataloader()))
    y = multimodal_model(batch)

    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model=multimodal_model, datamodule=sample_train_val_datamodule)
