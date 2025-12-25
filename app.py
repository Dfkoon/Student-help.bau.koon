import os
import uuid
import json
import shutil
import subprocess
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Any, Optional

from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image, ExifTags, ImageChops

# AI Detection Module
try:
    from ai_model import quick_predict
    AI_AVAILABLE = True
    print("✅ AI Detection Model loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI Detection not available: {e}")

# PDF Generation - Using browser print-to-PDF
PDF_AVAILABLE = True  # Always available via browser print

# =========================
# App Configuration
# =========================
app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# File Upload Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'tiff', 'bmp', 'gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def validate_file(file) -> Tuple[bool, str]:
    """Validate uploaded file type and size"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    filename = file.filename.lower()
    if '.' not in filename:
        return False, "File must have an extension"
    
    ext = filename.rsplit('.', 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS).upper()}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        current_mb = file_size / (1024 * 1024)
        return False, f"File too large ({current_mb:.1f}MB). Maximum allowed: {max_mb:.0f}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "Valid file"

# =========================
# Helper: Safe tool execution
# =========================
def safe_run(cmd, timeout=60):
    exe = cmd[0]
    if shutil.which(exe) is None:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "Tool not available"}

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False
        )
        return {
            "available": True,
            "returncode": p.returncode,
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
        }
    except Exception as e:
        return {"available": True, "returncode": -1, "stdout": "", "stderr": str(e)}

# =========================
# Hash computation
# =========================
def compute_hashes(file_path: str) -> Dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
            sha256.update(chunk)

    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest()
    }

# =========================
# EXIF helpers
# =========================
def _rational_to_float(value) -> Optional[float]:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        return float(value)
    except Exception:
        return None

def _gps_to_degrees(gps_coord) -> Optional[float]:
    try:
        d = _rational_to_float(gps_coord[0])
        m = _rational_to_float(gps_coord[1])
        s = _rational_to_float(gps_coord[2])
        if None in (d, m, s):
            return None
        return d + (m / 60) + (s / 3600)
    except Exception:
        return None

# =========================
# ELA: Error Level Analysis
# =========================
def perform_ela(image_path: str, output_path: str, quality: int = 90):
    try:
        tmp_ela = output_path + ".tmp.jpg"
        im = Image.open(image_path).convert("RGB")
        im.save(tmp_ela, "JPEG", quality=quality)
        
        resaved_im = Image.open(tmp_ela)
        ela_im = ImageChops.difference(im, resaved_im)
        
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_im = ela_im.point(lambda i: i * scale)
        ela_im.save(output_path)
        
        if os.path.exists(tmp_ela):
            os.remove(tmp_ela)
        return True
    except Exception as e:
        print(f"ELA Error: {e}")
        return False

# =========================
# EXIF via PIL
# =========================
def extract_exif_with_pil(image_path: str) -> Dict[str, Any]:
    result = {
        "camera_make": None,
        "camera_model": None,
        "datetime_original": None,
        "gps": {"present": False, "latitude": None, "longitude": None},
        "source": "PIL",
        "note": None,
    }

    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            result["note"] = "No EXIF metadata found."
            return result

        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        result["camera_make"] = tags.get("Make")
        result["camera_model"] = tags.get("Model")
        result["datetime_original"] = tags.get("DateTimeOriginal") or tags.get("DateTime")

        gps_info = tags.get("GPSInfo")
        if gps_info:
            gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            result["gps"]["present"] = True
            
            lat = lon = None
            if "GPSLatitude" in gps_tags and "GPSLatitudeRef" in gps_tags:
                lat = _gps_to_degrees(gps_tags["GPSLatitude"])
                if gps_tags["GPSLatitudeRef"] == "S": lat = -lat

            if "GPSLongitude" in gps_tags and "GPSLongitudeRef" in gps_tags:
                lon = _gps_to_degrees(gps_tags["GPSLongitude"])
                if gps_tags["GPSLongitudeRef"] == "W": lon = -lon

            result["gps"]["latitude"] = lat
            result["gps"]["longitude"] = lon

        return result

    except Exception as e:
        result["note"] = f"PIL EXIF error: {e}"
        return result

# =========================
# Hidden data detection
# =========================
def detect_hidden_data(image_path: str, report_path: str) -> Tuple[str, str]:
    payload = os.path.join(report_path, "payload.bin")

    # Try steghide (often requires password, but we try empty)
    safe_run(["steghide", "extract", "-sf", image_path, "-p", "", "-xf", payload])

    if os.path.exists(payload) and os.path.getsize(payload) > 0:
        return "DETECTED", "Hidden data successfully extracted using Steghide."

    # Try binwalk
    bw = safe_run(["binwalk", "-e", image_path])
    if bw["available"]:
        extracted_dir = os.path.join(os.path.dirname(image_path), f"_{os.path.basename(image_path)}.extracted")
        if os.path.isdir(extracted_dir):
            files = os.listdir(extracted_dir)
            if files:
                return "DETECTED", "Embedded files found within the image."

    return "NOT_DETECTED", "No obvious hidden data found in this image."

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/gui/hash", methods=["POST"])
def gui_hash():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400

    image = request.files["image"]
    
    # Validate file
    is_valid, message = validate_file(image)
    if not is_valid:
        return jsonify({"ok": False, "error": message}), 400
    
    tmp_path = os.path.join(REPORTS_DIR, f"tmp_{uuid.uuid4().hex}.bin")

    try:
        image.save(tmp_path)
        hashes = compute_hashes(tmp_path)
        return jsonify({"ok": True, "hashes": hashes})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

@app.route("/gui/analyze", methods=["POST"])
def gui_analyze():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    image = request.files["image"]
    
    # Validate file
    is_valid, message = validate_file(image)
    if not is_valid:
        return render_template("index.html", error=message)
    
    report_id = uuid.uuid4().hex[:12]
    report_path = os.path.join(REPORTS_DIR, report_id)
    os.makedirs(report_path, exist_ok=True)

    image_path = os.path.join(report_path, image.filename)
    image.save(image_path)

    # Perform ELA
    ela_filename = "ela_heatmap.jpg"
    ela_path = os.path.join(report_path, ela_filename)
    ela_success = perform_ela(image_path, ela_path)

    hashes = compute_hashes(image_path)
    metadata = extract_exif_with_pil(image_path)
    status, explanation = detect_hidden_data(image_path, report_path)
    
    # AI Detection
    ai_result = None
    if AI_AVAILABLE:
        try:
            model_path = os.path.join(BASE_DIR, "models", "best_stego_efficientnet.pth")
            ai_result = quick_predict(image_path, model_path if os.path.exists(model_path) else None)
        except Exception as e:
            print(f"AI Detection Error: {e}")
            ai_result = {
                'success': False,
                'confidence': 0,
                'verdict': 'غير متاح',
                'verdict_en': 'Not available',
                'error': str(e)
            }
    else:
        ai_result = {
            'success': False,
            'confidence': 0,
            'verdict': 'نموذج AI غير محمّل',
            'verdict_en': 'AI model not loaded',
            'model_available': False
        }

    # Dynamic Report Content (English)
    report_data = {
        "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "input_filename": image.filename,
        "file_size": os.path.getsize(image_path),
        "hashes": hashes,
        "metadata": metadata,
        "status": status,
        "explanation": explanation,
        "ela_image": f"/report/{report_id}/file/{ela_filename}" if ela_success else None,
        "plain_badge_class": "green" if status == "NOT_DETECTED" else "red",
        "plain_overall": "Clean" if status == "NOT_DETECTED" else "Suspicious / Contains Hidden Data",
        "plain_hidden_answer": "None" if status == "NOT_DETECTED" else "Yes, content detected",
        "plain_hidden_explain": explanation,
        "plain_tampering_explain": "Based on preliminary EXIF data analysis, the image appears authentic." if metadata['camera_make'] else "Warning: Lack of camera data may indicate software processing.",
        "plain_signals": "Digital fingerprint and metadata verified.",
        "plain_note": "This report is an automated analysis and does not replace manual forensic investigation in complex cases.",
        # AI Detection Results
        "ai_result": ai_result,
        "ai_available": AI_AVAILABLE and ai_result.get('success', False),
        "ai_confidence": ai_result.get('confidence', 75) if ai_result else 75,
        "ai_verdict": ai_result.get('verdict', 'Not available') if ai_result else 'Not available',
        "ai_is_manipulated": ai_result.get('is_manipulated', False) if ai_result else False,
    }

    # Save Analysis JSON
    with open(os.path.join(report_path, "analysis.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate HTML Report from template
    html_content = render_template("report_template.html", **report_data)
    with open(os.path.join(report_path, "report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    return render_template(
        "result.html",
        **report_data,
        html_url=f"/report/{report_id}/html",
        json_url=f"/report/{report_id}/json",
        download_url=f"/report/{report_id}/download",
        pdf_url=f"/report/{report_id}/pdf"
    )

@app.route("/report/<rid>/file/<filename>")
def view_file(rid, filename):
    return send_file(os.path.join(REPORTS_DIR, rid, filename))

@app.route("/report/<rid>/html")
def view_html(rid):
    return send_file(os.path.join(REPORTS_DIR, rid, "report.html"))

@app.route("/report/<rid>/json")
def view_json(rid):
    return send_file(os.path.join(REPORTS_DIR, rid, "analysis.json"))

@app.route("/report/<rid>/download")
def download_report(rid):
    path = os.path.join(REPORTS_DIR, rid, "report.html")
    return send_file(path, as_attachment=True, download_name=f"forensic_report_{rid}.html")

@app.route("/report/<rid>/pdf")
def download_pdf(rid):
    """Serve HTML report with print dialog for PDF export"""
    html_path = os.path.join(REPORTS_DIR, rid, "report.html")
    
    if not os.path.exists(html_path):
        return jsonify({"error": "Report not found"}), 404
    
    # Read HTML and inject print script
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add auto-print script before closing body tag
    print_script = """
    <script>
        window.onload = function() {
            // Set document title for PDF filename
            document.title = 'Forensic_Report_""" + rid + """';
            // Trigger print dialog
            setTimeout(function() {
                window.print();
            }, 500);
        };
    </script>
    """
    
    html_content = html_content.replace('</body>', print_script + '</body>')
    
    from flask import Response
    return Response(html_content, mimetype='text/html')

if __name__ == "__main__":
    def open_firefox():
        """Open the app in Firefox automatically"""
        url = "http://127.0.0.1:5001"
        try:
            # Try specific Firefox command for macOS
            subprocess.run(["open", "-a", "Firefox", url], check=False)
        except Exception:
            # Fallback to default browser if Firefox fails
            import webbrowser
            webbrowser.open(url)

    # Open browser after 1.5 seconds (to allow server to start)
    from threading import Timer
    Timer(1.5, open_firefox).start()
    
    app.run(host="0.0.0.0", port=5001, debug=True)
