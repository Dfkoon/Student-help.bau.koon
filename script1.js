// عناصر الصفحة
const casesContainer = document.getElementById('cases-container');
const caseDetailsDiv = document.getElementById('case-details');
const tabs = document.querySelectorAll('.tab');
const allCounter = document.getElementById('all-counter');
const murderCounter = document.getElementById('murder-counter');
const theftCounter = document.getElementById('theft-counter');
const mysteryCounter = document.getElementById('mystery-counter');
const solvedCountSpan = document.getElementById('solved-count');
const totalCountSpan = document.getElementById('total-count');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const solutionModal = document.getElementById('solution-modal');
const solutionInput = document.getElementById('solution-input');
const submitSolutionBtn = document.getElementById('submit-solution');
const cancelSolutionBtn = document.getElementById('cancel-solution');
const noticeBox = document.querySelector('.notice-box');

let currentCategory = 'all';
let currentCaseIndex = null;
let firstVisit = true;

// حساب عدد القضايا لكل تصنيف
function updateCounters() {
    allCounter.textContent = complexCases.length;
    murderCounter.textContent = complexCases.filter(c => c.category === 'murder').length;
    theftCounter.textContent = complexCases.filter(c => c.category === 'theft').length;
    mysteryCounter.textContent = complexCases.filter(c => c.category === 'mystery').length;
    document.getElementById('arson-counter').textContent = complexCases.filter(c => c.solved).length;
    totalCountSpan.textContent = complexCases.length;
    solvedCountSpan.textContent = complexCases.filter(c => c.solved).length;
    updateProgressBar();
    // إظهار صندوق الشهادة فقط عند حل جميع القضايا
    if (complexCases.length > 0 && complexCases.every(c => c.solved)) {
        noticeBox.style.display = '';
        setTimeout(() => {
            noticeBox.scrollIntoView({behavior: 'smooth'});
        }, 600);
    } else {
        noticeBox.style.display = 'none';
    }
}

// تحديث شريط التقدم
function updateProgressBar() {
    const solved = complexCases.filter(c => c.solved).length;
    const total = complexCases.length;
    const percent = total ? Math.round((solved / total) * 100) : 0;
    progressFill.style.width = percent + '%';
    progressText.textContent = percent + '%';
}

// عرض القضايا حسب التصنيف
function renderCases() {
    casesContainer.innerHTML = '';
    let filteredCases = complexCases;
    if (currentCategory === 'arson') {
        filteredCases = complexCases.filter(c => c.solved);
    } else if (currentCategory !== 'all') {
        filteredCases = complexCases.filter(c => c.category === currentCategory);
    }
    filteredCases.forEach((caseItem, index) => {
        const card = document.createElement('div');
        card.className = 'case-card' + (caseItem.solved ? ' solved' : '');
        card.innerHTML = `
            <div class="case-header">
                <span class="case-number">${index + 1}</span>
                <h3>${caseItem.title}</h3>
            </div>
            <div class="case-body">
                <div class="case-description">${caseItem.description}</div>
                <div class="case-stats">
                    <div class="stat">
                        <i class="fas fa-fingerprint"></i>
                        <span>${caseItem.evidence.length} أدلة</span>
                    </div>
                    <div class="stat">
                        <i class="fas fa-users"></i>
                        <span>${caseItem.suspects.length} مشتبه</span>
                    </div>
                    <div class="stat">
                        <i class="fas fa-brain"></i>
                        <span>صعوبة: ${caseItem.complexity}/10</span>
                    </div>
                </div>
                <a href="#" class="read-more" data-idx="${complexCases.indexOf(caseItem)}">
                    <i class="fas fa-search"></i> تحقق في القضية
                </a>

            </div>
        `;
        casesContainer.appendChild(card);
    });
}

// عرض تفاصيل القضية
function showCaseDetails(idx) {
    const c = complexCases[idx];
    const solvedClass = c.solved ? 'solved-green' : '';
    caseDetailsDiv.innerHTML = `
        <div class="case-details ${solvedClass}" style="display:block;">
            <button class="back-button"><i class="fas fa-arrow-right"></i> رجوع للقائمة</button>
            <h2 class="case-title">${c.title}</h2>
            <div class="case-description">${c.description}</div>
            <div class="complexity-badge">الصعوبة: ${c.complexity}/10</div>
            <div class="evidence-list">
                <h3><i class="fas fa-flask"></i> الأدلة</h3>
                ${c.evidence.map(e => `<div class="evidence-item${e.important ? ' important' : ''}">${e.text}</div>`).join('')}
            </div>
            <div class="suspects-list">
                <h3><i class="fas fa-user-secret"></i> المشتبهون</h3>
                <div class="suspects-grid">
                    ${c.suspects.map(s => `
                        <div class="suspect-card">
                            <h4>${s.name}</h4>
                            <p>${s.description}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            ${c.solved ? `<div class="solved-box" style="background:#27ae60;color:#fff;font-weight:bold;text-align:center;margin:2rem 0;padding:1.2rem;border-radius:10px;box-shadow:0 2px 12px #27ae6055;">تم حل القضية! ✅</div>` : `
                <button class="solve-case" data-idx="${idx}"><i class="fas fa-gavel"></i> حل القضية</button>
            `}
        </div>
    `;
    caseDetailsDiv.scrollIntoView({ behavior: 'smooth' });
    currentCaseIndex = idx;
    // زر الرجوع
    caseDetailsDiv.querySelector('.back-button').onclick = () => {
        caseDetailsDiv.innerHTML = '';
    };
    // زر حل القضية
    const solveBtn = caseDetailsDiv.querySelector('.solve-case');
    if (solveBtn) {
        solveBtn.onclick = () => {
            solutionModal.style.display = 'block';
            solutionInput.value = '';
            solutionInput.focus();
        };
    }
}

// تفعيل الفلاتر (tabs)
tabs.forEach(tab => {
    tab.onclick = () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentCategory = tab.getAttribute('data-category');
        noticeBox.style.display = 'none';
        renderCases();
        caseDetailsDiv.innerHTML = '';
    };
});

// تفعيل زر تفاصيل القضية
casesContainer.addEventListener('click', function (e) {
    const btn = e.target.closest('.read-more');
    if (btn) {
        e.preventDefault();
        const idx = parseInt(btn.getAttribute('data-idx'));
        showCaseDetails(idx);
    }
});

// نافذة الحل: زر الإلغاء
cancelSolutionBtn.onclick = () => {
    solutionModal.style.display = 'none';
};

// نافذة الحل: زر التأكيد
submitSolutionBtn.onclick = () => {
    const answer = solutionInput.value.trim();
    if (!answer) {
        solutionInput.style.borderColor = 'var(--warning-color)';
        solutionInput.placeholder = 'يرجى كتابة اسم المشتبه!';
        return;
    }
    const c = complexCases[currentCaseIndex];
    // تحقق من الحل (case-insensitive)
    const correct = c.solution.some(sol =>
        sol.replace(/\s/g, '').toLowerCase() === answer.replace(/\s/g, '').toLowerCase()
    );
    if (correct) {
        c.solved = true;
        solutionModal.style.display = 'none';
        showCaseDetails(currentCaseIndex);
        updateCounters();
        // رسالة نجاح
        setTimeout(() => {
            alert('أحسنت! تم حل القضية بنجاح ✅');
        }, 300);
    } else {
        solutionInput.style.borderColor = 'var(--warning-color)';
        solutionInput.value = '';
        solutionInput.placeholder = 'الإجابة غير صحيحة، حاول مرة أخرى!';
    }
};

// إغلاق نافذة الحل عند الضغط خارجها
window.onclick = function (e) {
    if (e.target === solutionModal) {
        solutionModal.style.display = 'none';
    }
};

// إظهار صندوق الشهادة فقط عند أول دخول (مع إخفاء القضايا)، ثم إخفاؤه عند فتح أي قسم، وإظهاره مجدداً فقط عند حل جميع القضايا.
function toggleNoticeBox() {
    if (!noticeBox) return;
    if (firstVisit) {
        noticeBox.style.display = '';
        casesContainer.innerHTML = '';
        caseDetailsDiv.innerHTML = '';
        firstVisit = false;
        return;
    }
    // إظهار الصندوق فقط عند حل جميع القضايا
    if (complexCases.length > 0 && complexCases.every(c => c.solved)) {
        noticeBox.style.display = '';
        setTimeout(() => {
            noticeBox.scrollIntoView({behavior: 'smooth'});
        }, 600);
    } else {
        noticeBox.style.display = 'none';
    }
}
window.addEventListener('DOMContentLoaded', toggleNoticeBox);

// تهيئة الصفحة
updateCounters();
renderCases();