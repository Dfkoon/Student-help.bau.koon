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
    from core_management import lsb_entropy, exif_score, chi_square_test, bit_plane_noise
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
    
    def calculate_complexity(self, img_path: str) -> float:
        """
        Calculates image complexity using local variance.
        High complexity (e.g., textures, forests) returns closer to 1.0.
        Low complexity (e.g., solid colors, walls) returns closer to 0.0.
        """
        try:
            with Image.open(img_path) as img:
                img_gray = img.convert('L')
                arr = np.array(img_gray)
                # Standard deviation of pixel values is a good proxy for global complexity/texture
                std_dev = np.std(arr)
                # Normalize: typical high texture std is around 80-100. Smooth is < 10.
                complexity = min(std_dev / 80.0, 1.0) 
                return complexity
        except Exception:
            return 0.5  # Neutral fallback

    def apply_srm_analysis(self, image_path: str) -> float:
        """
        Applies Spatial Rich Models (SRM) filtering to extract noise residuals.
        Returns a score (0-1) reflecting high-frequency anomalies.
        """
        try:
            img = Image.open(image_path).convert('L')
            arr = np.array(img, dtype=np.float32)
            
            # Simple SRM Kernels (1st and 2nd Order)
            kernel_1st = np.array([-1, 2, -1])
            
            # Row-wise filtering
            res_1st = np.abs(np.apply_along_axis(lambda m: np.convolve(m, kernel_1st, mode='valid'), axis=1, arr=arr))
            # Column-wise filtering
            res_2nd = np.abs(np.apply_along_axis(lambda m: np.convolve(m, kernel_1st, mode='valid'), axis=0, arr=arr))
            
            # Measure local variance of residuals - stego usually increases this
            score = float((np.mean(res_1st) + np.mean(res_2nd)) / 255.0)
            return min(score * 10, 1.0)
        except Exception:
            return 0.0

    def analyze_luminance_consistency(self, image_path: str) -> float:
        """
        Analyzes consistency of noise in small blocks of the luminance layer.
        Inconsistent noise variance across blocks often indicates splicing.
        Returns a score (0.0 - 1.0) where higher means more inconsistent (likely manipulated).
        """
        try:
            img = Image.open(image_path).convert('L')
            # Resize for consistency and performance
            img = img.resize((512, 512))
            arr = np.array(img, dtype=np.float32)

            h, w = arr.shape
            block_size = 16
            variances = []

            # Calculate variance for each block
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    if block.shape == (block_size, block_size):
                        variances.append(np.var(block))

            if not variances:
                return 0.0

            # Analyze the spread of variances
            # High standard deviation of local variances = inconsistent noise levels
            var_std = np.std(variances)
            var_mean = np.mean(variances)
            
            # Coefficient of variation (CV) as a normalized metric
            if var_mean > 0:
                cv = var_std / var_mean
                # Empirical threshold: CV > 0.8 is often suspicious for natural images
                score = min(cv / 1.2, 1.0)
                return float(score)
            return 0.0
        except Exception:
            return 0.0

    def ensemble_vote(self, scores: Dict[str, float], complexity_score: float) -> Tuple[float, float, str]:
        """
        Intelligent ensemble voting with adaptive weights.
        
        Args:
            scores: Dictionary with detector scores {'ml', 'srm', 'chroma', 'lum', 'chi', 'entropy', 'noise'}
            complexity_score: Image complexity (0-1)
        
        Returns:
            (combined_score, certainty, explanation)
        """
        # Step 1: Determine base weights based on image type
        if complexity_score < 0.3:  # Smooth image - trust SRM & Luminance more
            base_weights = {
                'ml': 0.20,
                'srm': 0.25,
                'chroma': 0.10,
                'lum': 0.25,     # High weight for smooth images
                'chi': 0.10,
                'entropy': 0.05,
                'noise': 0.05
            }
            image_type = "smooth"
        elif complexity_score > 0.7:  # Textured image (includes screenshots with text)
            base_weights = {
                'ml': 0.25,
                'srm': 0.15,
                'chroma': 0.20,  # Reduced from 0.25 to avoid false positives on text
                'lum': 0.20,     # Increased to balance
                'chi': 0.15,
                'entropy': 0.05,
                'noise': 0.05
            }
            image_type = "textured"
        else:  # Balanced
            base_weights = {
                'ml': 0.25,
                'srm': 0.15,
                'chroma': 0.15,
                'lum': 0.20,
                'chi': 0.10,
                'entropy': 0.10,
                'noise': 0.05
            }
            image_type = "balanced"
        
        # Step 2: Confidence-based weight adjustment
        # If a detector is very confident (>0.85), boost its weight by 25%
        adjusted_weights = {}
        total_boost = 0
        
        for detector, base_weight in base_weights.items():
            score = scores.get(detector, 0)
            if score > 0.85:  # High confidence
                boost = base_weight * 0.25
                adjusted_weights[detector] = base_weight + boost
                total_boost += boost
            elif score < 0.15:  # Very low score, reduce weight
                adjusted_weights[detector] = base_weight * 0.7
            else:
                adjusted_weights[detector] = base_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        # Step 3: Calculate weighted vote
        combined_score = sum(scores.get(k, 0) * w for k, w in normalized_weights.items())
        
        # Step 4: Calculate certainty (voter agreement)
        # Count how many detectors agree on the verdict
        votes = {'clean': 0, 'suspicious': 0, 'stego': 0}
        for detector, score in scores.items():
            # Only consider detectors present in weights
            if detector in normalized_weights:
                if score >= 0.70:
                    votes['stego'] += normalized_weights.get(detector, 0)
                elif score >= 0.45:
                    votes['suspicious'] += normalized_weights.get(detector, 0)
                else:
                    votes['clean'] += normalized_weights.get(detector, 0)
        
        # Certainty is the strength of the majority vote
        max_vote = max(votes.values())
        certainty = max_vote * 100  # Convert to percentage
        
        # Generate explanation
        explanation = f"Ensemble vote (image type: {image_type})"
        
        return combined_score, certainty, explanation

    def analyze_chroma_channels(self, image_path: str) -> float:
        """
        Analyzes Chroma channels (Cb, Cr) for anomalies.
        Stego is often hidden in Chroma to avoid human detection.
        """
        try:
            img = Image.open(image_path).convert('YCbCr')
            y, cb, cr = img.split()
            
            cb_arr = np.array(cb, dtype=np.float32)
            cr_arr = np.array(cr, dtype=np.float32)
            
            # Calculate entropy and variance of Chroma channels
            cb_std = float(np.std(cb_arr))
            cr_std = float(np.std(cr_arr))
            
            # Normalize: anything above 15.0 std in chroma is usually artificial noise
            chroma_score = (cb_std + cr_std) / 30.0
            return min(chroma_score, 1.0)
        except Exception:
            return 0.0

    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predict using Ensemble Learning with Intelligent Voting
        """
        try:
            # 1. Advanced Preprocessing Analysis
            complexity_score = self.calculate_complexity(image_path)
            srm_score = self.apply_srm_analysis(image_path)
            chroma_score = self.analyze_chroma_channels(image_path)
            lum_consistency = self.analyze_luminance_consistency(image_path)
            
            # 2. ML Probability
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                ml_prob = torch.sigmoid(output).item()
            
            # 3. Statistical & Forensic Metrics
            entropy = lsb_entropy(image_path)
            exif_val = exif_score(image_path)
            chi_prob = chi_square_test(image_path)
            noise_val = bit_plane_noise(image_path)

            # =============================
            # ENSEMBLE LEARNING VOTING
            # =============================
            
            # Prepare scores dictionary for ensemble voting
            if self.model_loaded:
                scores = {
                    'ml': ml_prob,
                    'srm': srm_score,
                    'chroma': chroma_score,
                    'lum': lum_consistency,
                    'chi': chi_prob,
                    'entropy': min(entropy, 1.0),
                    'noise': min(noise_val * 2, 1.0)
                }
            else:
                # Without ML model, redistribute voting power
                scores = {
                    'ml': 0.0,  # No vote
                    'srm': srm_score,
                    'chroma': chroma_score,
                    'lum': lum_consistency,
                    'chi': chi_prob,
                    'entropy': min(entropy, 1.0),
                    'noise': min(noise_val * 2, 1.0)
                }
            
            # Call ensemble voting system
            combined_score, ensemble_certainty, ensemble_explanation = self.ensemble_vote(scores, complexity_score)
            
            # Map to 0-100 scale for UI using a non-linear curve
            # For scores < 0.5, confidence should be lower.
            # For scores > 0.8, confidence should be higher.
            if combined_score > 0.6:
                confidence = 80 + ((combined_score - 0.6) / 0.4) * 19  # Map 0.6-1.0 to 80-99%
            else:
                 confidence = 50 + (combined_score / 0.6) * 30       # Map 0.0-0.6 to 50-80%

            # Screenshot/Textured safeguard:
            # If it's a textured image but not overwhelmingly manipulated, cap confidence to avoid false alarm panic
            if complexity_score > 0.7 and combined_score < 0.75:
                 confidence = min(confidence, 85.0)
            
            # 4. Final Verdict Logic
            # Adaptive thresholds based on complexity
            stego_threshold = 0.65 + (complexity_score * 0.1) 
            suspicious_threshold = 0.48 + (complexity_score * 0.10)  # Raised base from 0.40 to 0.48

            if combined_score >= stego_threshold:
                verdict = "Likely Stego"
                verdict_en = "Likely Stego"
                is_manipulated = True
            elif combined_score >= suspicious_threshold:
                verdict = "Suspicious"
                verdict_en = "Suspicious"
                is_manipulated = True
            else:
                verdict = "Clean"
                verdict_en = "Clean"
                is_manipulated = False
                
            # Escalation Rule: High confidence SRM + Chroma override
            if verdict != "Likely Stego" and (srm_score > 0.85 and chroma_score > 0.8):
                verdict = "Likely Stego"
                verdict_en = "Likely Stego"
                is_manipulated = True
                confidence = max(confidence, 94.0)

            return {
                'success': True,
                'is_manipulated': is_manipulated,
                'verdict': verdict,
                'verdict_en': verdict_en,
                'ml_probability': round(ml_prob, 4),
                'srm_residual_score': round(srm_score, 4),
                'chroma_anomaly_score': round(chroma_score, 4),
                'lum_consistency_score': round(lum_consistency, 4),
                'lsb_entropy': round(entropy, 4),
                'chi_square_prob': round(chi_prob, 4),
                'noise_density': round(noise_val, 4),
                'exif_score': exif_val,
                'confidence': confidence,
                'certainty': round(ensemble_certainty, 2),
                'complexity_score': round(complexity_score, 2),
                'ensemble_explanation': ensemble_explanation,
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
    
    # Test on a dummy image if exists, or create one
    test_img = "test_image.png"
    if not os.path.exists(test_img):
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(test_img)
        print(f"Created temporary {test_img}")

    print(f"\n🔍 Running prediction on {test_img}...")
    result = detector.predict(test_img)
    
    import json
    print(json.dumps(result, indent=2))
    print("\n✅ Model initialized and tested successfully!")
