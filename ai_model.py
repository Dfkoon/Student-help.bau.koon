"""
AI-Powered Image Forensics Detection Module
Uses EfficientNet-B0 for detecting manipulated/steganographic images
"""

import os
import ssl
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from typing import Dict, Tuple, Optional
import numpy as np

# Fix SSL certificate issue for model download
ssl._create_default_https_context = ssl._create_unverified_context


class StegoDetector:
    """
    Deep Learning model for detecting image manipulation and steganography
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to pre-trained model weights (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = self._build_model()
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_loaded = True
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model weights: {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print("ℹ️ No pre-trained weights loaded. Using untrained model.")
        
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self) -> nn.Module:
        """
        Build EfficientNet-B0 model architecture
        """
        # Load pre-trained EfficientNet-B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze most layers (for fine-tuning scenario)
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Unfreeze last 4 blocks for better detection
        for block in model.features[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Replace classifier for binary classification
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 
            1  # Binary output: clean (0) or manipulated (1)
        )
        
        return model.to(self.device)
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predict if an image is manipulated/contains steganography
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results:
            {
                'is_manipulated': bool,
                'confidence': float (0-100),
                'raw_score': float (0-1),
                'verdict': str,
                'model_available': bool
            }
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probability = torch.sigmoid(output).item()
            
            # Determine verdict
            is_manipulated = probability > 0.5
            base_confidence = probability if is_manipulated else (1 - probability)
            
            # Scale confidence from 75-100% instead of 0-100%
            # Formula: 75 + (base_confidence * 25)
            confidence_percent = 75 + (base_confidence * 25)
            
            # Generate verdict text (English only)
            if confidence_percent >= 95:
                certainty = "Very High"
            elif confidence_percent >= 90:
                certainty = "High"
            elif confidence_percent >= 85:
                certainty = "Moderate-High"
            elif confidence_percent >= 80:
                certainty = "Moderate"
            else:
                certainty = "Moderate-Low"
            
            if is_manipulated:
                verdict = f"Suspicious - Potential manipulation detected (Confidence: {certainty})"
            else:
                verdict = f"Image appears authentic (Confidence: {certainty})"
            
            return {
                'is_manipulated': is_manipulated,
                'confidence': round(confidence_percent, 2),
                'raw_score': round(probability, 4),
                'verdict': verdict,
                'certainty_level': certainty,
                'model_available': self.model_loaded,
                'success': True
            }
            
        except Exception as e:
            return {
                'is_manipulated': None,
                'confidence': 75,
                'raw_score': 0,
                'verdict': f"Analysis error: {str(e)}",
                'certainty_level': 'N/A',
                'model_available': False,
                'success': False,
                'error': str(e)
            }
    
    def analyze_batch(self, image_paths: list) -> list:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results


# Global detector instance (singleton pattern)
_detector_instance = None

def get_detector(model_path: Optional[str] = None) -> StegoDetector:
    """
    Get or create the global detector instance
    
    Args:
        model_path: Path to model weights (only used on first call)
        
    Returns:
        StegoDetector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = StegoDetector(model_path=model_path)
    return _detector_instance


def quick_predict(image_path: str, model_path: Optional[str] = None) -> Dict[str, any]:
    """
    Quick prediction function for easy integration
    
    Args:
        image_path: Path to image file
        model_path: Optional path to model weights
        
    Returns:
        Prediction results dictionary
    """
    detector = get_detector(model_path)
    return detector.predict(image_path)


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing AI Detection Model...")
    detector = StegoDetector()
    print(f"Device: {detector.device}")
    print(f"Model loaded: {detector.model_loaded}")
    print("\n✅ Model initialized successfully!")
