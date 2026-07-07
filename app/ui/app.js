const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const chatHistory = document.getElementById('chatHistory');
const sendBtn = document.getElementById('sendBtn');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');

// New brutalist elements
const urlInput = document.getElementById('urlInput');
const ingestUrlBtn = document.getElementById('ingestUrlBtn');
const urlStatus = document.getElementById('urlStatus');
const clearKbBtn = document.getElementById('clearKbBtn');

let chartInstances = {};

function triggerConfetti() {
    if (window.confetti) {
        window.confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
    }
}

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle enter to submit
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

function createMessageElement(content, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'ME' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    div.appendChild(avatar);
    div.appendChild(contentDiv);
    return div;
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    contentDiv.innerHTML = `
        <div class="thinking-container">
            <div class="thinking-step active" id="ts-1"><div class="spinner"></div> Retrieving context...</div>
            <div class="thinking-step" id="ts-2">Reranking chunks...</div>
            <div class="thinking-step" id="ts-3">Generating response...</div>
        </div>
    `;
    
    div.appendChild(avatar);
    div.appendChild(contentDiv);
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Simulate thinking steps
    setTimeout(() => {
        const t1 = document.getElementById('ts-1');
        const t2 = document.getElementById('ts-2');
        if(t1) { t1.classList.remove('active'); t1.classList.add('done'); t1.innerHTML = 'Context retrieved.'; }
        if(t2) { t2.classList.add('active'); t2.innerHTML = '<div class="spinner"></div> Reranking chunks...'; }
    }, 1500);

    setTimeout(() => {
        const t2 = document.getElementById('ts-2');
        const t3 = document.getElementById('ts-3');
        if(t2) { t2.classList.remove('active'); t2.classList.add('done'); t2.innerHTML = 'Chunks reranked.'; }
        if(t3) { t3.classList.add('active'); t3.innerHTML = '<div class="spinner"></div> Generating response...'; }
    }, 3000);
}

function removeTyping() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

async function typeWriterEffect(element, htmlContent) {
    element.innerHTML = htmlContent + '<span class="typewriter-cursor"></span>';
    const cursor = element.querySelector('.typewriter-cursor');
    setTimeout(() => {
        if(cursor) cursor.remove();
    }, 2000);
}

function getScoreClass(score) {
    if (score >= 0.8) return 'score-high';
    if (score >= 0.5) return 'score-med';
    return 'score-low';
}

function updateInsights(data) {
    if (!data.explainability) return;
    
    const exp = data.explainability;
    const latency = exp.latency_ms || {};
    
    let html = `
        <div class="metric-card">
            <h4>Latency Breakdown</h4>
            <div class="metric-row"><span>Retrieval</span><span class="metric-value">${latency.retrieval_ms || 0}ms</span></div>
            <div class="metric-row"><span>Reranking</span><span class="metric-value">${latency.reranking_ms || 0}ms</span></div>
            <div class="metric-row"><span>Generation</span><span class="metric-value">${latency.generation_ms || 0}ms</span></div>
            <div class="metric-row" style="margin-top: 12px; padding-top: 12px; border-top: 3px solid #000;">
                <strong>Total</strong><strong class="metric-value">${latency.total_ms || 0}ms</strong>
            </div>
        </div>

        <div class="metric-card">
            <h4>Pipeline Stats</h4>
            <div class="metric-row"><span>Retrieved Chunks</span><span class="metric-value">${exp.retrieved_chunks}</span></div>
            <div class="metric-row"><span>Reranked</span><span class="metric-value">${exp.reranked_chunks}</span></div>
            <div class="metric-row"><span>Total Tokens</span><span class="metric-value">${exp.total_tokens}</span></div>
            <div class="metric-row"><span>Confidence</span><span class="metric-value ${getScoreClass(data.confidence)}">${(data.confidence * 100).toFixed(0)}%</span></div>
        </div>
        
        <div class="metric-card">
            <h4>Retrieved Sources</h4>
    `;

    data.sources.forEach((src, idx) => {
        html += `
            <div class="citation" onclick="this.classList.toggle('open')">
                <div class="citation-header">
                    Source ${idx + 1}: ${src.filename || 'Unknown'}
                    <span>${(src.similarity_score * 100).toFixed(0)}% match</span>
                </div>
                <div class="citation-content">
                    ${src.content}
                </div>
            </div>
        `;
    });

    html += '</div>';
    document.getElementById('insightsContent').innerHTML = html;

    // Update charts
    if(chartInstances.latency) {
        chartInstances.latency.data.datasets[0].data = [
            latency.retrieval_ms || 0,
            latency.reranking_ms || 0,
            latency.generation_ms || 0
        ];
        chartInstances.latency.update();
    }
    
    if(chartInstances.tokens) {
        chartInstances.tokens.data.labels.push(new Date().toLocaleTimeString());
        chartInstances.tokens.data.datasets[0].data.push(exp.total_tokens || 0);
        if(chartInstances.tokens.data.labels.length > 10) {
            chartInstances.tokens.data.labels.shift();
            chartInstances.tokens.data.datasets[0].data.shift();
        }
        chartInstances.tokens.update();
    }
    
    if(chartInstances.confidence) {
        chartInstances.confidence.data.labels.push(new Date().toLocaleTimeString());
        chartInstances.confidence.data.datasets[0].data.push(data.confidence !== undefined ? Math.round(data.confidence * 100) : 0);
        if(chartInstances.confidence.data.labels.length > 10) {
            chartInstances.confidence.data.labels.shift();
            chartInstances.confidence.data.datasets[0].data.shift();
        }
        chartInstances.confidence.update();
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = userInput.value.trim();
    if (!query) return;

    // 1. Add User Message
    chatHistory.appendChild(createMessageElement(`<p>${query}</p>`, 'user'));
    
    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // 2. Show typing indicator
    showTyping();

    try {
        const response = await fetch('/api/v1/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();
        removeTyping();
        sendBtn.disabled = false;

        if (response.ok) {
            const htmlContent = marked.parse(data.answer);
            const msgEl = createMessageElement('', 'assistant');
            chatHistory.appendChild(msgEl);
            const contentDiv = msgEl.querySelector('.message-content');
            typeWriterEffect(contentDiv, htmlContent);
            
            updateInsights(data);
        } else {
            throw new Error(data.detail || 'Failed to get answer');
        }

    } catch (error) {
        removeTyping();
        sendBtn.disabled = false;
        
        chatHistory.appendChild(createMessageElement(
            `<p style="color: #FF5D8F;"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${error.message}</p>`, 
            'assistant'
        ));
    }
    
    chatHistory.scrollTop = chatHistory.scrollHeight;
});


// Upload functionality
if(uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--text-main)';
        uploadArea.style.background = '#f0f0f0';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#000';
        uploadArea.style.background = 'var(--bg-primary)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#000';
        uploadArea.style.background = 'var(--bg-primary)';
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadStatus.innerHTML = `<div class="spinner" style="display:inline-block; vertical-align:middle; margin-right:8px;"></div> Uploading & Ingesting...`;
    uploadStatus.style.color = '#000';

    try {
        const response = await fetch('/api/v1/documents/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.innerHTML = `<span style="color: #00E676;">✔ ${data.message} (${data.stats.chunks_stored || 0} chunks)</span>`;
            if (typeof confetti === 'function') {
                confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
            }
            fetchDocumentsList();
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        uploadStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${error.message}`;
        uploadStatus.style.color = 'var(--accent)';
    }
}

// URL Ingestion
if(ingestUrlBtn && urlInput) {
    ingestUrlBtn.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        if(!url) return;
        
        urlStatus.innerHTML = `<div class="spinner" style="display:inline-block; vertical-align:middle; margin-right:8px;"></div> Scraping URL...`;
        urlStatus.style.color = '#000';
        
        try {
            const response = await fetch('/api/v1/documents/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });
            const data = await response.json();
            
            if (response.ok) {
                urlStatus.innerHTML = `<span style="color: #00E676;">✔ ${data.message} (${data.stats?.chunks_stored || 0} chunks)</span>`;
                urlInput.value = '';
                if (typeof confetti === 'function') {
                    confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
                }
                fetchDocumentsList();
            } else {
                throw new Error(data.detail || 'Failed to ingest URL');
            }
        } catch (error) {
            urlStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${error.message}`;
            urlStatus.style.color = 'var(--accent)';
        }
    });
}

// Clear KB
if(clearKbBtn) {
    clearKbBtn.addEventListener('click', async () => {
        if(!confirm("Are you sure? This will wipe the vector database entirely!")) return;
        
        clearKbBtn.innerHTML = "Wiping...";
        
        try {
            const response = await fetch('/api/v1/documents/clear', { method: 'POST' });
            if (response.ok) {
                clearKbBtn.innerHTML = "Wiped! ✔️";
                clearKbBtn.style.background = "#00E676";
                if (typeof confetti === 'function') {
                    confetti({ particleCount: 150, spread: 100, origin: { y: 0.6 } });
                }
                setTimeout(() => {
                    clearKbBtn.innerHTML = "Clear Knowledge Base";
                    clearKbBtn.style.background = "var(--danger)";
                }, 3000);
                fetchDocumentsList();
            } else {
                alert("Failed to clear DB.");
                clearKbBtn.innerHTML = "Clear Knowledge Base";
            }
        } catch(e) {
            alert(e);
            clearKbBtn.innerHTML = "Clear Knowledge Base";
        }
    });
}


// View switching logic
function switchView(viewId) {
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });

    const targetPanel = document.getElementById(viewId);
    if(targetPanel) {
        targetPanel.classList.add('active');
    }
    
    // Find nav item by data target
    const navItem = document.querySelector(`.nav-item[data-target="${viewId}"]`);
    if(navItem) navItem.classList.add('active');
}

// Init Charts on load
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchDocumentsList();
});

function initCharts() {
    const latCtx = document.getElementById('latencyChart');
    const tokCtx = document.getElementById('tokenChart');
    const confCtx = document.getElementById('confidenceChart');
    
    if(!latCtx || !tokCtx || !confCtx) return;

    // Brutalist chart styling
    Chart.defaults.font.family = "'Space Mono', monospace";
    Chart.defaults.color = '#000';
    
    chartInstances.latency = new Chart(latCtx, {
        type: 'radar',
        data: {
            labels: ['Retrieval', 'Reranking', 'Generation'],
            datasets: [{
                label: 'Latency (ms)',
                data: [0, 0, 0],
                backgroundColor: 'rgba(255, 93, 143, 0.4)', // Neon Pink transparent
                borderColor: '#000',
                borderWidth: 3,
                pointBackgroundColor: '#FFDE00',
                pointBorderColor: '#000',
                pointBorderWidth: 2
            }]
        },
        options: { scales: { r: { grid: { color: 'rgba(0,0,0,0.2)' }, pointLabels: { color: '#000', font: { weight: 'bold' } }, ticks: { display: false } } }, plugins: { legend: { display: false } } }
    });

    chartInstances.tokens = new Chart(tokCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Tokens',
                data: [],
                backgroundColor: '#00E5FF', // Cyan
                borderColor: '#000',
                borderWidth: 3,
                borderRadius: 0
            }]
        },
        options: { scales: { y: { grid: { color: 'rgba(0,0,0,0.1)' } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } }
    });

    chartInstances.confidence = new Chart(confCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Confidence %',
                data: [],
                borderColor: '#00E676', // Neon green
                tension: 0, // Brutalist harsh lines (no tension)
                borderWidth: 3,
                fill: true,
                backgroundColor: 'rgba(0, 230, 118, 0.2)',
                pointBackgroundColor: '#fff',
                pointBorderColor: '#000'
            }]
        },
        options: { scales: { y: { min: 0, max: 100, grid: { color: 'rgba(0,0,0,0.1)' } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } }
    });
}

// Fetch and display ingested materials
async function fetchDocumentsList() {
    const listEl = document.getElementById('materialsList');
    if (!listEl) return;
    
    try {
        const response = await fetch('/api/v1/documents/list');
        const data = await response.json();
        
        if (data.status === 'success') {
            if (data.sources && data.sources.length > 0) {
                let html = '';
                data.sources.forEach(src => {
                    html += `
                        <div style="padding: 12px; border: 3px solid #000; background: #fff; box-shadow: 2px 2px 0px 0px #000;">
                            <strong>${src.file_name}</strong><br>
                            <span style="font-size: 0.85em; color: #555;">Type: ${src.type} | Format: ${src.format}</span>
                        </div>`;
                });
                listEl.innerHTML = html;
            } else {
                listEl.innerHTML = `<div style="padding: 10px; border: 2px solid #000; background: #fff;">No documents found.</div>`;
            }
        }
    } catch (e) {
        listEl.innerHTML = `<div style="padding: 10px; border: 2px solid #FF5D8F; background: #fff; color: #FF5D8F;">Failed to load materials</div>`;
    }
}
