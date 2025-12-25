"""
Training Script for Steganography Detection Model
Optional - Use this to train your own model with custom dataset
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse


class StegoDataset(Dataset):
    """Dataset for loading cover and stego images"""
    
    def __init__(self, folder_path, transform=None):
        self.files = []
        self.labels = []

        for label_name, label in [('cover', 0), ('stego', 1)]:
            dir_path = os.path.join(folder_path, label_name)
            if not os.path.exists(dir_path):
                print(f"⚠ Warning: {dir_path} not found!")
                continue

            for f in os.listdir(dir_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.files.append(os.path.join(dir_path, f))
                    self.labels.append(label)

        self.transform = transform
        print(f"📁 Loaded {len(self.files)} images ({label_name})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def train_model(data_dir, output_path="models/best_stego_efficientnet.pth", epochs=10, batch_size=16):
    """
    Train the steganography detection model
    
    Args:
        data_dir: Root directory containing train/val/test folders
        output_path: Where to save the best model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    
    # Create models directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print("📊 Loading datasets...")
    train_dataset = StegoDataset(os.path.join(data_dir, "train"), transform)
    val_dataset = StegoDataset(os.path.join(data_dir, "val"), transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Build model
    print("🏗️ Building model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Unfreeze last 4 blocks
    for block in model.features[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    
    # Replace classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100
        print(f"📈 Train Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = torch.sigmoid(model(imgs))
                preds = (outputs > 0.5).float()
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = (val_correct / val_total) * 100
        print(f"✅ Validation Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"✨ Best model saved! (Accuracy: {val_acc:.2f}%)")
    
    print(f"\n{'='*50}")
    print(f"🏆 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Steganography Detection Model")
    parser.add_argument("--data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="models/best_stego_efficientnet.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    print("🎯 Starting Model Training...")
    print(f"Data directory: {args.data}")
    print(f"Output path: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    train_model(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
