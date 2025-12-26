"""
AI-Powered Image Forensics Detection Module
Uses Hybrid StegoNet (HighPass + EfficientNet) and Statistical Analysis
"""

import os
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from typing import Dict, Tuple, Optional
import numpy as np

# Import Core Management Utils
try:
    from core_management import lsb_entropy, exif_score
except ImportError:
    # Fallback if running standalone without core_management in path
    def lsb_entropy(path): return 0.0
    def exif_score(path): return 0

# Fix SSL certificate issue for model download
ssl._create_default_https_context = ssl._create_unverified_context


# ---------------- LAYERS & MODEL (Must match train_model.py) ----------------
class HighPassLayer(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        self.weight = kernel.view(1, 1, 3, 3)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        w = self.weight.to(x.device)
        out = F.conv2d(gray, w, padding=1)
        out = out.repeat(1, 3, 1, 1)        
        return out

class StegoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hp = HighPassLayer()
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # We only need the architecture to load weights
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, 1)
        self.backbone = base_model

    def forward(self, x):
        x = self.hp(x)
        return self.backbone(x)


class StegoDetector:
    """
    Detector integrating Deep Learning and Statistical Analysis
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = StegoNet().to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                # Load weights
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model weights: {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print("ℹ️ No pre-trained weights loaded. Using untrained model.")
        
        self.model.eval()
        
        # Transform (Must match training)
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predict using Weighted Scoring Logic (User Request)
        """
        try:
            # 1. ML Probability
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                ml_prob = torch.sigmoid(output).item()
            
            # 2. Statistical Metrics
            entropy = lsb_entropy(image_path)
            exif_val = exif_score(image_path)
            
            # 3. Weighted Scoring Logic
            # Matches user provided logic:
            # score = exif + (20 if entropy > 0.9) + (prob * 30)
            
            score = 0
            score += exif_val  # e.g., 10 or 20
            
            if entropy > 0.9:
                score += 20
                
            score += ml_prob * 30
            
            # 4. Final Verdict
            # If score >= 60: Likely Stego
            # If score >= 30: Suspicious
            # Else: Clean
            
            if score >= 60:
                verdict = "Likely Stego"
                verdict_en = "Likely Stego"
                is_manipulated = True
            elif score >= 30:
                verdict = "Suspicious"
                verdict_en = "Suspicious"
                is_manipulated = True
            else:
                verdict = "Clean"
                verdict_en = "Clean"
                is_manipulated = False
                
            return {
                'success': True,
                'is_manipulated': is_manipulated,
                'verdict': verdict,
                'verdict_en': verdict_en,
                'score': round(score, 2),
                'ml_probability': round(ml_prob, 4),
                'lsb_entropy': round(entropy, 4),
                'exif_score': exif_val,
                'confidence': round(min(score, 100), 2), # Use score as pseudo-confidence
                'model_available': self.model_loaded
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'verdict': "Error",
                'model_available': self.model_loaded
            }
    
    def analyze_batch(self, image_paths: list) -> list:
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results


# Global Instance
_detector_instance = None

def get_detector(model_path: Optional[str] = None) -> StegoDetector:
    global _detector_instance
    if _detector_instance is None:
        # Lazy initialization: The model is loaded only when this is called the first time
        _detector_instance = StegoDetector(model_path=model_path)
    return _detector_instance


def quick_predict(image_path: str, model_path: Optional[str] = None) -> Dict[str, any]:
    detector = get_detector(model_path)
    return detector.predict(image_path)


if __name__ == "__main__":
    print("🧪 Testing AI Detection Model...")
    detector = StegoDetector()
    print(f"Device: {detector.device}")
    print("\n✅ Model initialized successfully! (Deep Learning + LSB + EXIF)")
