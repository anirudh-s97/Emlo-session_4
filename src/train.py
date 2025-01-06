import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from datamodule import DogBreedsDataModule
from model import DogBreedClassifier
from rich import print

def main():
    print("[bold blue]Starting training...[/bold blue]")
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize data module
    dm = DogBreedsDataModule()
    dm.prepare_data()
    # Initialize model
    model = DogBreedClassifier(
        *dm.dims,
        dm.num_classes,
        learning_rate=1e-3
    )
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",  # Updated save path
        filename="dog-breed-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min"
    )
    
    # Initialize logger
    logger = TensorBoardLogger("logs", name="dog_breeds")
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        logger=logger,
    )
    
    # Train and test the model
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == "__main__":
    main()