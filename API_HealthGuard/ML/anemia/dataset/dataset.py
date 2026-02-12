import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

class AnemiaDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, augment=False):
        """
        Dataset India avec structure Number/images
        
        Args:
            csv_path: Path vers train_labels.csv
            root_dir: API/ML/data/India/
            transform: Transformations custom (optionnel)
            augment: Augmentation data pour training
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        
        # Transformations
        if transform:
            self.transform = transform
        elif augment:
            # Training: Augmentation forte (dataset petit)
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.3),  # Yeux orientation variable
                transforms.ColorJitter(
                    brightness=0.3, 
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test: Pas d'augmentation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Path image
        img_path = self.root_dir / row['image_path']
        
        # Charger image
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"❌ Erreur chargement {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Labels et metadata
        label = torch.tensor(row['anemia_label'], dtype=torch.long)
        hgb = torch.tensor(row['hgb'], dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'hgb': hgb,
            'patient': row['patient_number'],
            'path': str(img_path)
        }
