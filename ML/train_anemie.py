import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path


from dataset.dataset import AnemiaDataset


# Configuration optimisée
CONFIG = {
    'data_dir': 'ML/data/',
    'train_csv': 'ML/data/train_combined.csv',
    'val_csv': 'ML/data/val_combined.csv',
    'model_name': 'efficientnet_b0',
    'batch_size': 16,
    'epochs': 80,             
    'lr': 3e-5,                
    'weight_decay': 5e-4,      
    'patience': 15,          
    'dropout': 0.3,           
    'class_weight_boost': 2.0, 
    'save_dir': 'models/'
}


def train_epoch(model, loader, criterion, optimizer, device):
    """Training epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation avec métriques complètes"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(loader, desc='Validation'):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Métriques
    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss, val_acc, precision, recall, f1, cm


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    # Datasets
    train_dataset = AnemiaDataset(
        CONFIG['train_csv'],
        CONFIG['data_dir'],
        augment=True
    )
    val_dataset = AnemiaDataset(
        CONFIG['val_csv'],
        CONFIG['data_dir'],
        augment=False
    )
    
    print(f"✅ Train: {len(train_dataset)} images")
    print(f"✅ Val: {len(val_dataset)} images")
    
    # ✅ Calcul des class weights
    train_df = pd.read_csv(CONFIG['train_csv'])
    class_counts = train_df['anemia_label'].value_counts().sort_index().values
    
    print(f"\n📊 Distribution train:")
    print(f"   Non-anémie (0): {class_counts[0]} images")
    print(f"   Anémie (1): {class_counts[1]} images")
    
    # Poids pour pénaliser les faux négatifs
    total = len(train_df)
    class_weights = torch.FloatTensor([
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1]) * CONFIG['class_weight_boost']
    ]).to(device)
    
    print(f"⚖️  Class weights: [Non-anémie: {class_weights[0]:.3f}, Anémie: {class_weights[1]:.3f}]")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=True,
        num_classes=2,
        drop_rate=CONFIG['dropout']  # Dropout sur classifier
    )
    
    print(f"\n🔒 Freeze layers...")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'blocks.6' not in name and 'classifier' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    total_params = frozen_params + trainable_params
    print(f"   Frozen: {frozen_params:,} params")
    print(f"   Trainable: {trainable_params:,} params ({100*trainable_params/total_params:.1f}%)")
    
    model = model.to(device)
    print(f"✅ Modèle: {CONFIG['model_name']} (avec Dropout {CONFIG['dropout']})")
    
    # ✅ Loss avec class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (seulement les params trainables)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5
    )
    
    # Track previous LR
    prev_lr = CONFIG['lr']
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    Path(CONFIG['save_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔥 Training démarré: {CONFIG['epochs']} epochs")
    print(f"   LR: {CONFIG['lr']:.6f} | Weight Decay: {CONFIG['weight_decay']:.6f}")
    print(f"   Batch Size: {CONFIG['batch_size']} | Patience: {CONFIG['patience']}\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = validate(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0,0]:2d} FP={cm[0,1]:2d}   (Non-anémie)")
        print(f"    FN={cm[1,0]:2d} TP={cm[1,1]:2d}   (Anémie)")
        
        # ✅ Overfitting warning
        if train_acc - val_acc > 0.15:
            print(f"  ⚠️  Overfitting détecté: écart {(train_acc - val_acc)*100:.1f}%")
        
        # LR scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr != prev_lr:
            print(f"  📉 LR réduit: {prev_lr:.6f} → {current_lr:.6f}")
            prev_lr = current_lr
        else:
            print(f"  📊 Learning Rate: {current_lr:.6f}")
        
        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            checkpoint_path = Path(CONFIG['save_dir']) / 'model_anemie.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_recall': val_rec,
                'config': CONFIG
            }, checkpoint_path)
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Recall: {val_rec:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{CONFIG['patience']}")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠️ Early stopping (patience={CONFIG['patience']})")
            break
    
    print(f"\n{'='*70}")
    print(f"🎉 Training terminé!")
    print(f"{'='*70}")
    print(f"🏆 Best Val Acc: {best_val_acc:.4f}")
    print(f"🏆 Best Val F1: {best_val_f1:.4f}")
    print(f"💾 Modèle: {CONFIG['save_dir']}/best_model.pt")


if __name__ == '__main__':
    main()
