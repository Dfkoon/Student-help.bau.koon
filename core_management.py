from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np

# ---------- ELA (Error Level Analysis) ----------
def generate_ela(image_path, quality=90):
    """
    Generates an Error Level Analysis (ELA) image.
    ELA highlights areas where the compression level differs from the rest of the image.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        temp_path = image_path + '.ela.tmp.jpg'
        original.save(temp_path, 'JPEG', quality=quality)

        compressed = Image.open(temp_path)
        diff = ImageChops.difference(original, compressed)

        extrema = diff.getextrema()
        max_diff = max([e[1] for e in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1

        ela_image = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Clean up temp file is handled by caller or OS usually, but strictly we should clean here
        # For performance in heavy loops we might keep it in memory, but here file IO is safe
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return ela_image
    except Exception as e:
        print(f"Error generating ELA: {e}")
        return None


# ---------- LSB Entropy (Least Significant Bit) ----------
def lsb_entropy(image_path):
    """
    Calculates the entropy of the Least Significant Bits (LSB).
    High entropy (> 0.9 approx) can indicate hidden encrypted data.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        # Extract LSB
        lsb = pixels & 1
        
        # Flatten to 1D array
        lsb_flat = lsb.flatten()
        
        # Calculate probabilities
        values, counts = np.unique(lsb_flat, return_counts=True)
        probs = counts / counts.sum()

        # Shannon Entropy
        # We use a small epsilon to avoid log2(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        return float(entropy)
    except Exception as e:
        print(f"Error calculating LSB entropy: {e}")
        return 0.0


# ---------- EXIF Score ----------
def exif_score(image_path):
    """
    Calculates a suspicion score based on EXIF metadata.
    Missing EXIF or software markers increase the score.
    """
    try:
        img = Image.open(image_path)
        exif = img.getexif()

        if not exif:
            return 10  # suspicious: missing EXIF suggests editing or stripping

        score = 0
        tag_names = {}
        for tag, value in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            tag_names[decoded] = value

        # Check for editing software signatures
        software_keys = ['Software', 'ProcessingSoftware']
        for key in software_keys:
            if key in tag_names:
                score += 10
                # Could add checks for "Adobe", "GIMP", etc. if needed
        
        return score
    except Exception as e:
        print(f"Error scoring EXIF: {e}")
        return 0

def chi_square_test(image_path: str) -> float:
    """
    Classic Chi-Square test for LSB steganography.
    Analyzes 'Pairs of Values' (PoV) distribution.
    Returns a probability (0-1) where higher = more likely to have hidden data.
    """
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert('L')
        pixels = np.array(img).flatten()
        
        # Count frequencies of each pixel value
        counts = np.zeros(256)
        for p in pixels:
            counts[p] += 1
            
        chi_sq = 0
        df = 0
        
        # Analyze pairs (0,1), (2,3) ... (254,255)
        for i in range(0, 256, 2):
            y_obs = counts[i]
            y_exp = (counts[i] + counts[i+1]) / 2.0
            
            if y_exp > 0:
                chi_sq += ((y_obs - y_exp) ** 2) / y_exp
                df += 1
        
        if df == 0:
            return 0.0
            
        # Simplified probability estimation (higher chi_sq = higher probability of stego)
        # In professional steganography, we'd use the incomplete gamma function, 
        # but a normalized score works well for this heuristic.
        prob = min(chi_sq / (df * 10), 1.0) # Normalizing roughly
        return float(prob)
        
    except Exception as e:
        print(f"Chi-Square error: {e}")
        return 0.0

def bit_plane_noise(image_path: str) -> float:
    """Analyze high-frequency noise in the LSB plane."""
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert('L')
        pixels = np.array(img)
        lsb_plane = pixels & 1
        
        # Calculate horizontal and vertical transitions
        h_diff = np.abs(lsb_plane[:, :-1] - lsb_plane[:, 1:])
        v_diff = np.abs(lsb_plane[:-1, :] - lsb_plane[1:, :])
        
        noise_density = (np.sum(h_diff) + np.sum(v_diff)) / (lsb_plane.size * 2)
        return float(noise_density)
    except:
        return 0.0
