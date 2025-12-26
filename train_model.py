"""
Training Script for Steganography Detection Model
Includes High Pass Filter Layer (SRM-like) and EfficientNet Backbone
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse


# ---------------- DATASET ----------------
class StegoDataset(Dataset):
    """Dataset for loading cover and stego images"""
    
    def __init__(self, folder_path, transform=None):
        self.files = []
        self.labels = []

        # Check for both standard structure and flat structure
        if os.path.exists(os.path.join(folder_path, 'cover')) and os.path.exists(os.path.join(folder_path, 'stego')):
            encoded_labels = [('cover', 0), ('stego', 1)]
        else:
            # Fallback or custom logic if needed, but for now expect standard structure
            encoded_labels = [('cover', 0), ('stego', 1)]

        for label_name, label in encoded_labels:
            dir_path = os.path.join(folder_path, label_name)
            if not os.path.exists(dir_path):
                print(f"⚠ Warning: {dir_path} not found!")
                continue

            for f in os.listdir(dir_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.files.append(os.path.join(dir_path, f))
                    self.labels.append(label)

        self.transform = transform
        print(f"📁 Loaded {len(self.files)} images from {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # Return a dummy tensor or handle gracefully (here we rely on robust data)
            return torch.zeros((3, 192, 192)), torch.tensor(0.0)


# ---------------- LAYERS & MODEL ----------------
class HighPassLayer(nn.Module):
    """
    SRM-like High Pass Filter Layer to extract residuals (noise features)
    Essential for detecting steganographic modifications
    """
    def __init__(self):
        super().__init__()
        # SRM filter kernel (KV kernel usually, or simple high pass)
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        self.weight = kernel.view(1, 1, 3, 3)

    def forward(self, x):
        # Convert RGB to Gray for filtering
        # shape: [B, 3, H, W] -> [B, 1, H, W]
        gray = x.mean(dim=1, keepdim=True) 
        
        # Apply convolution
        w = self.weight.to(x.device)
        out = F.conv2d(gray, w, padding=1)
        
        # Replicate to 3 channels so backbone (EfficientNet) accepts it
        out = out.repeat(1, 3, 1, 1)        
        return out


class StegoNet(nn.Module):
    """
    Hybrid model: HighPassLayer + EfficientNet-B0
    """
    def __init__(self):
        super().__init__()
        self.hp = HighPassLayer()
        
        # Load backbone
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze initial layers of backbone
        for p in base_model.features.parameters():
            p.requires_grad = False
            
        # Unfreeze last blocks for fine-tuning
        for block in base_model.features[-4:]:
            for p in block.parameters():
                p.requires_grad = True
                
        # Custom classifier head
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, 1)
        self.backbone = base_model

    def forward(self, x):
        x = self.hp(x)
        return self.backbone(x)


# ---------------- TRAINING FUNCTION ----------------
def train_model(data_dir, output_path="models/best_stego_model.pth", epochs=10, batch_size=16):
    
    # Create models directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Transforms (Avoid destructive transforms like JPEG compression)
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load Data
    print("📊 Loading datasets...")
    train_dataset = StegoDataset(os.path.join(data_dir, "train"), transform)
    val_dataset = StegoDataset(os.path.join(data_dir, "val"), transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ Error: No training data found. Please check directory structure.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Model Setup
    print("🏗️ Building StegoNet (HighPass + EfficientNet)...")
    model = StegoNet().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )
    
    # Training Loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\nExample Epoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_acc = (correct / total) * 100 if total > 0 else 0
        print(f"📈 Train Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
        print(f"✅ Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"✨ Best model saved to {output_path}")

    print(f"\n🏆 Final Best Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Steganography Detection Model")
    parser.add_argument("--data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="models/best_stego_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.data, args.output, args.epochs, args.batch_size)
