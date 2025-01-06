import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import subprocess


class DogBreedsDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Dog Breeds dataset.
    Handles downloading, extracting, and organizing the dataset, as well as creating dataloaders.
    """
    def __init__(
        self, 
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dims = (3, 224, 224)
        self.num_classes = 120

    def _install_gdown(self):
        """Install gdown package using pip"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("Successfully installed gdown")
        except subprocess.CalledProcessError:
            print("Failed to install gdown. Please install it manually using: pip install gdown")
            sys.exit(1)

    def _download_dataset(self):
        """Download the dataset using gdown"""
        try:
            import gdown
        except ImportError:
            self._install_gdown()
            import gdown

        zip_path = self.data_dir / "dog_breeds.zip"
        url = "https://drive.google.com/uc?id=17YKP-SjAKpG5f8wUKdS-4f6Wfi_bbr6f"
        
        print("Downloading dataset...")
        gdown.download(url, str(zip_path), quiet=False)
        return zip_path

    def _extract_dataset(self, zip_path: Path):
        """Extract the dataset using zipfile"""
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Print total files to extract
                total_files = len(zip_ref.namelist())
                print(f"Total files to extract: {total_files}")
                
                # Extract all files with progress
                for i, file in enumerate(zip_ref.namelist(), 1):
                    zip_ref.extract(file, self.data_dir)
                    if i % 100 == 0:  # Print progress every 100 files
                        print(f"Extracted {i}/{total_files} files")
                
            print("Extraction completed successfully!")
            
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file")
            raise
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            raise
        finally:
            # Clean up zip file
            if zip_path.exists():
                zip_path.unlink()

    def _organize_dataset(self):
        """Organize the dataset into train and test splits"""
        print("Organizing dataset...")
        print(self.data_dir)
        # Find the Images directory
        potential_source_dirs = list(self.data_dir.glob("**/dataset"))
        if not potential_source_dirs:
            raise FileNotFoundError("Could not find the Images directory in the extracted dataset")
        
        source_dir = potential_source_dirs[0]
        train_dir = self.data_dir / "train"
        test_dir = self.data_dir / "test"
        
        # Create directories
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Get all breed directories
        breed_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        
        if not breed_dirs:
            raise FileNotFoundError(f"No breed directories found in {source_dir}")
        
        print(f"Found {len(breed_dirs)} breed directories")
        
        # Process each breed
        for breed_dir in breed_dirs:
            breed_name = breed_dir.name
            print(f"Processing {breed_name}")
            
            # Create breed directories
            (train_dir / breed_name).mkdir(exist_ok=True)
            (test_dir / breed_name).mkdir(exist_ok=True)
            
            # Get all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
                image_files.extend(list(breed_dir.glob(ext)))
            
            if not image_files:
                print(f"Warning: No images found in {breed_name}")
                continue
                
            # Split images
            split_idx = int(len(image_files) * 0.8)
            
            # Copy to train
            for img_file in image_files[:split_idx]:
                shutil.copy2(str(img_file), str(train_dir / breed_name / img_file.name))
            
            # Copy to test
            for img_file in image_files[split_idx:]:
                shutil.copy2(str(img_file), str(test_dir / breed_name / img_file.name))
        
        # Clean up
        extracted_dir = source_dir.parent
        if extracted_dir.exists() and extracted_dir.name != self.data_dir.name:
            shutil.rmtree(str(extracted_dir))

    def prepare_data(self):
        """Download and prepare the dataset"""
        try:
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True)
            
            train_dir = self.data_dir / "train"
            test_dir = self.data_dir / "test"
            
            # Only download and organize if necessary
            if not (train_dir.exists() and test_dir.exists() and 
                   any(train_dir.iterdir()) and any(test_dir.iterdir())):
                zip_path = self._download_dataset()
                self._extract_dataset(zip_path)
                self._organize_dataset()
                print("Dataset preparation completed!")
                print(f"Train directory: {train_dir}")
                print(f"Test directory: {test_dir}")
            else:
                print("Dataset already prepared.")
            
        except Exception as e:
            print(f"Error during dataset preparation: {str(e)}")
            raise

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets"""
        if stage == "fit" or stage is None:
            self.data_train = ImageFolder(
                root=str(self.data_dir / "train"),
                transform=self.transform
            )
            self.data_val = ImageFolder(
                root=str(self.data_dir / "test"),
                transform=self.transform
            )
            self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    # Example usage
    datamodule = DogBreedsDataModule()
    datamodule.prepare_data()  # This will download and organize the dataset