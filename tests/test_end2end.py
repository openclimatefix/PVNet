import lightning
import torch
from torch.utils.data import Dataset, DataLoader


def test_model_trainer_fit(multimodal_model, sample_batch):
    """Test end-to-end training with solar coords included."""
    
    # Verification that sample batch works with model
    print("Keys in batch:", sample_batch.keys())
    if "solar_azimuth" in sample_batch:
        print("solar_azimuth shape:", sample_batch["solar_azimuth"].shape)
    if "solar_elevation" in sample_batch:
        print("solar_elevation shape:", sample_batch["solar_elevation"].shape)
    
    # Check forward pass
    y = multimodal_model(sample_batch)
    print(f"Forward pass output shape: {y.shape}")
    
    # Simple Dataset - sample_batch
    class SimpleDataset(Dataset):
        def __len__(self):
            return 10
            
        def __getitem__(self, idx):
            return idx
    
    dataloader = DataLoader(
        SimpleDataset(),
        batch_size=1,
        collate_fn=lambda x: sample_batch
    )
    
    # Lightning DataModule
    class SimpleDataModule(lightning.LightningDataModule):
        def train_dataloader(self):
            return dataloader
            
        def val_dataloader(self):
            return dataloader
    
    # Train with Lightning
    trainer = lightning.pytorch.trainer.trainer.Trainer(
        fast_dev_run=True,
        accelerator="cpu"
    )
    
    trainer.fit(model=multimodal_model, datamodule=SimpleDataModule())
