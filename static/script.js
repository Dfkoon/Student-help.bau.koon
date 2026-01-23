// Clock Functionality
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    const dateStr = now.toLocaleDateString('en-US', {
        weekday: 'long',
        day: 'numeric',
        month: 'long',
        year: 'numeric'
    });

    document.getElementById('clock').textContent = timeStr;
    document.getElementById('date').textContent = dateStr;
}
setInterval(updateClock, 1000);
updateClock();

// File Selection Feedback
const fileInput = document.getElementById('imageInput');
const fileNameDisplay = document.getElementById('fileName');
const dropZone = document.getElementById('dropZone');

// Drag & Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = 'var(--primary)', false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = 'var(--border)', false);
});

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    fileInput.files = files;
    if (files.length > 0) {
        validateAndDisplayFile(files[0]);
    }
}

// Client-side file validation
function validateAndDisplayFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp', 'image/gif'];
    const maxSize = 50 * 1024 * 1024; // 50 MB

    // Check file type
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file (JPG, PNG, WEBP, TIFF, BMP, GIF).');
        fileInput.value = '';
        fileNameDisplay.textContent = 'No file chosen';
        return false;
    }

    // Check file size
    if (file.size > maxSize) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        showError(`File too large (${sizeMB}MB). Maximum allowed: 50MB`);
        fileInput.value = '';
        fileNameDisplay.textContent = 'No file chosen';
        return false;
    }

    // File is valid
    fileNameDisplay.textContent = file.name;
    dropZone.style.borderColor = 'var(--primary)';
    return true;
}

function showError(message) {
    // Remove existing error if any
    const existingError = document.getElementById('errorBanner');
    if (existingError) {
        existingError.remove();
    }

    // Create new error banner
    const errorBanner = document.createElement('div');
    errorBanner.className = 'error-banner';
    errorBanner.id = 'errorBanner';
    errorBanner.innerHTML = `
        <span class="error-icon">⚠️</span>
        <span class="error-text">${message}</span>
        <button class="error-close" onclick="closeError()">✕</button>
    `;

    // Insert at the beginning of the card
    const card = document.querySelector('.card');
    card.insertBefore(errorBanner, card.firstChild);
}

function closeError() {
    const errorBanner = document.getElementById('errorBanner');
    if (errorBanner) {
        errorBanner.style.animation = 'slideUp 0.3s ease-out';
        setTimeout(() => errorBanner.remove(), 300);
    }
}

// Update file input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        validateAndDisplayFile(e.target.files[0]);
    }
});

// Hash Logic
let currentHashes = null;

function checkHash() {
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const hashBtn = document.getElementById('hashBtn');
    const spinner = document.getElementById('hashSpinner');

    hashBtn.disabled = true;
    spinner.style.display = 'inline-block';

    const formData = new FormData();
    formData.append("image", file);

    fetch("/gui/hash", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (!data.ok) {
                alert(data.error || "Hash calculation failed");
                return;
            }

            currentHashes = data.hashes;
            document.getElementById("md5Value").textContent = data.hashes.md5;
            document.getElementById("sha256Value").textContent = data.hashes.sha256;
            document.getElementById("hashBox").style.display = "block";

            // Scroll to hash box
            document.getElementById("hashBox").scrollIntoView({ behavior: 'smooth' });
        })
        .catch(() => {
            alert("Failed to calculate hash");
        })
        .finally(() => {
            hashBtn.disabled = false;
            spinner.style.display = 'none';
        });
}

function downloadHashes() {
    if (!currentHashes) return;

    const text = `STELLAR FORENSICS HASH REPORT\n` +
        `Generated: ${new Date().toISOString()}\n` +
        `File: ${fileInput.files[0].name}\n` +
        `--------------------------------------------\n` +
        `MD5: ${currentHashes.md5}\n` +
        `SHA-256: ${currentHashes.sha256}\n` +
        `--------------------------------------------`;

    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hash_report_${fileInput.files[0].name}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// TrueFocus Component Logic
class TrueFocus {
    constructor(options) {
        this.container = document.getElementById(options.id);
        this.sentence = options.sentence || "";
        this.blurAmount = options.blurAmount || 5;
        this.borderColor = options.borderColor || "red";
        this.animationDuration = options.animationDuration || 2;
        this.pauseBetweenAnimations = options.pauseBetweenAnimations || 1;
        this.words = [];
        this.currentIndex = 0;

        this.init();
    }

    init() {
        this.container.style.setProperty('--blur-amount', `${this.blurAmount}px`);
        this.container.style.setProperty('--focus-border-color', this.borderColor);

        const wordsArr = this.sentence.split(" ");
        wordsArr.forEach((wordText, index) => {
            const span = document.createElement('span');
            span.className = 'true-focus-word';
            span.textContent = wordText;
            this.container.appendChild(span);
            this.words.push(span);
        });

        this.animate();
    }

    animate() {
        // Focus current word
        this.words.forEach(w => w.classList.remove('focused'));
        this.words[this.currentIndex].classList.add('focused');

        // Move to next word or restart
        this.currentIndex = (this.currentIndex + 1) % this.words.length;

        // Determine timing: if it wrapped around, add the extra pause
        const isLastWord = this.currentIndex === 0;
        const nextStepDelay = isLastWord
            ? (this.animationDuration + this.pauseBetweenAnimations) * 1000
            : this.animationDuration * 1000;

        setTimeout(() => this.animate(), nextStepDelay);
    }
}

// Initialize TrueFocus with User Parameters
new TrueFocus({
    id: 'trueFocus',
    sentence: "True Focus Forensics",
    blurAmount: 5,
    borderColor: "#818cf8",
    animationDuration: 2,
    pauseBetweenAnimations: 1
});

// Dynamic Loading Animation Logic
const loadingMessages = [
    "Initializing Secure Environment...",
    "Loading Neural Network Weights...",
    "Extracting Bit-Plane Layers...",
    "Extracting Luminance Layer...",
    "Calculating Shannon Entropy...",
    "Applying SRM Filters...",
    "Analyzing Chroma Channels...",
    "Performing Chi-Square Analysis...",
    "Validating Cryptographic Signatures...",
    "Generating Final Forensic Report..."
];

function showLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
        const statusText = document.getElementById('loadingStatus');
        const progressBar = document.getElementById('progressBar');
        const progressPercentage = document.getElementById('progressPercentage');

        if (statusText && progressBar && progressPercentage) {
            let step = 0;
            let progress = 0;

            // Show first message immediately
            statusText.textContent = loadingMessages[0];
            progressBar.style.width = '0%';
            progressPercentage.textContent = '0%';

            // Update message every 1.5 seconds
            setInterval(() => {
                step = (step + 1) % loadingMessages.length;
                statusText.textContent = loadingMessages[step];
            }, 1500);

            // Animate progress bar from 0% to 95% over ~12 seconds
            setInterval(() => {
                if (progress < 95) {
                    progress += Math.random() * 8; // Random increment for realistic feel
                    progress = Math.min(progress, 95); // Cap at 95%
                    progressBar.style.width = progress + '%';
                    progressPercentage.textContent = Math.floor(progress) + '%';
                }
            }, 400);
        }
    }
}

// Attach Loading Screen to Form Submission
document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', (e) => {
            // Basic validation check before showing loader
            const fileInput = document.getElementById('imageInput');
            if (fileInput && fileInput.files.length > 0) {
                showLoading();
            }
        });
    }
});
