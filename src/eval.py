import pytorch_lightning as pl
from datamodule import DogBreedsDataModule
from model import DogBreedClassifier
import torch


def evaluate():
    data_module = DogBreedsDataModule('./data')  # Ensure data path is correct
    checkpoint_path = '/app/checkpoints/dogs_classifier-best_val_loss.ckpt'
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path, strict=False)
   

    # Use accelerator='gpu' or accelerator='auto' instead of gpus
    trainer = pl.Trainer(accelerator='auto')
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    evaluate()


