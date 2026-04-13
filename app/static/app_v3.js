/**
 * TB Detector v3 - Integrated Pipeline JavaScript
 * Complete workflow: Upload → Preprocess → Train → Visualize → Save → Predict
 */

const API_URL = window.location.origin;

// Global State
let currentSection = 'upload';
let datasetUploaded = false;
let preprocessed = false;
let trainingInProgress = false;
let trainingCompleted = false;
let predictionAudio = null;
let statusPollInterval = null;
let logsPollInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('TB Detector v3 Initialized');
    checkPipelineStatus();
});

// ========== SECTION NAVIGATION ==========

function goToSection(section) {
    // Update section visibility
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(`section-${section}`).classList.add('active');
    
    // Update step indicators
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'completed', 'error');
    });
    
    const sections = ['upload', 'preprocess', 'train', 'results', 'predict'];
    const currentIdx = sections.indexOf(section);
    
    for (let i = 0; i < sections.length; i++) {
        const stepEl = document.getElementById(`step-${sections[i]}`);
        if (i < currentIdx) {
            stepEl.classList.add('completed');
        } else if (i === currentIdx) {
            stepEl.classList.add('active');
        }
    }
    
    // Update progress bar
    const progress = ((currentIdx + 1) / sections.length) * 100;
    document.getElementById('pipelineProgress').style.width = `${progress}%`;
    
    currentSection = section;
    
    // Section specific actions
    if (section === 'results') {
        loadTrainingResults();
    } else if (section === 'preprocess') {
        startStatusPolling();
    } else if (section === 'train') {
        startStatusPolling();
    } else if (section === 'predict') {
        refreshModelList();
    }
}

// ========== DATASET UPLOAD ==========

function handleZipUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.zip')) {
        showAlert('uploadAlert', 'File harus berformat ZIP', 'error');
        return;
    }
    
    uploadDataset(file);
}

async function uploadDataset(file) {
    showLoading('Uploading dataset...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('extract_only', 'false');
        
        const response = await fetch(`${API_URL}/dataset/upload`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (result.success) {
            datasetUploaded = true;
            
            // Show dataset info
            document.getElementById('uploadArea').classList.add('success');
            document.getElementById('datasetInfo').classList.remove('hidden');
            document.getElementById('statAudioCount').textContent = result.audio_count;
            document.getElementById('statStatus').textContent = 'Uploaded';
            document.getElementById('statPath').textContent = 'OK';
            
            showAlert('uploadAlert', `Dataset berhasil diupload: ${result.audio_count} file audio`, 'success');
            
            // Auto advance to preprocessing if auto_preprocess
            if (result.auto_preprocess) {
                setTimeout(() => goToSection('preprocess'), 1500);
            }
        } else {
            showAlert('uploadAlert', result.message || 'Upload failed', 'error');
        }
        
    } catch (err) {
        hideLoading();
        showAlert('uploadAlert', 'Error: ' + err.message, 'error');
    }
}

function resetDataset() {
    document.getElementById('uploadArea').classList.remove('success');
    document.getElementById('datasetInfo').classList.add('hidden');
    document.getElementById('zipInput').value = '';
    datasetUploaded = false;
}

// ========== PREPROCESSING ==========

async function startPreprocessing() {
    if (!datasetUploaded) {
        showAlert('preprocessAlert', 'Dataset belum diupload!', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/dataset/preprocess`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('preprocessAlert', 'Preprocessing dimulai...', 'success');
            startStatusPolling();
            startLogsPolling('preprocess');
        } else {
            showAlert('preprocessAlert', result.message, 'error');
        }
        
    } catch (err) {
        showAlert('preprocessAlert', 'Error: ' + err.message, 'error');
    }
}

// ========== TRAINING ==========

function toggleBackbone(element) {
    element.classList.toggle('selected');
    const checkbox = element.querySelector('input');
    checkbox.checked = element.classList.contains('selected');
}

// Make backbone options clickable
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.backbone-option').forEach(option => {
        option.addEventListener('click', () => toggleBackbone(option));
    });
});

async function startTraining() {
    if (!preprocessed && !datasetUploaded) {
        showAlert('trainAlert', 'Dataset belum siap!', 'warning');
        return;
    }
    
    // Get selected backbones
    const selectedBackbones = [];
    document.querySelectorAll('.backbone-option.selected input').forEach(cb => {
        selectedBackbones.push(cb.value);
    });
    
    if (selectedBackbones.length === 0) {
        showAlert('trainAlert', 'Pilih minimal satu backbone!', 'warning');
        return;
    }
    
    const config = {
        backbones: selectedBackbones,
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batchSize').value),
        learning_rate: parseFloat(document.getElementById('learningRate').value),
        patience: parseInt(document.getElementById('patience').value),
        augment: true,
        pos_weight: 2.0
    };
    
    try {
        document.getElementById('trainConfig').classList.add('hidden');
        document.getElementById('trainProgress').classList.remove('hidden');
        document.getElementById('btnStartTrain').disabled = true;
        
        const response = await fetch(`${API_URL}/training/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (result.success) {
            trainingInProgress = true;
            showAlert('trainAlert', 'Training dimulai...', 'success');
            startStatusPolling();
            startLogsPolling('train');
        } else {
            showAlert('trainAlert', result.message, 'error');
            document.getElementById('trainConfig').classList.remove('hidden');
            document.getElementById('trainProgress').classList.add('hidden');
            document.getElementById('btnStartTrain').disabled = false;
        }
        
    } catch (err) {
        showAlert('trainAlert', 'Error: ' + err.message, 'error');
        document.getElementById('trainConfig').classList.remove('hidden');
        document.getElementById('trainProgress').classList.add('hidden');
        document.getElementById('btnStartTrain').disabled = false;
    }
}

// ========== RESULTS & VISUALIZATION ==========

async function loadTrainingResults() {
    try {
        const response = await fetch(`${API_URL}/training/results`);
        const data = await response.json();

        if (data.training_completed && data.models.length > 0) {
            trainingCompleted = true;

            // Show chart
            const chartImg = document.getElementById('comparisonChart');
            const noChartMsg = document.getElementById('noChartMessage');

            chartImg.src = `${API_URL}/training/visualization?t=${Date.now()}`;
            chartImg.style.display = 'block';
            noChartMsg.style.display = 'none';

            // Render model cards
            renderModelCards(data.results);

            // Generate download buttons
            generateDownloadButtons(data.results);
        } else {
            document.getElementById('noChartMessage').textContent =
                'Training belum selesai atau tidak ada model yang dilatih.';
        }

    } catch (err) {
        console.error('Error loading results:', err);
    }
}

function generateDownloadButtons(results) {
    const container = document.getElementById('downloadButtons');
    container.innerHTML = '';

    Object.keys(results).forEach(modelName => {
        const btn = document.createElement('button');
        btn.className = 'btn btn-secondary';
        btn.style.padding = '10px 20px';
        btn.style.fontSize = '0.9rem';
        btn.innerHTML = `📥 ${formatModelName(modelName)}`;
        btn.onclick = () => downloadModel(modelName);
        container.appendChild(btn);
    });
}

async function downloadModel(modelName) {
    try {
        showLoading('Mempersiapkan download...');

        const response = await fetch(`${API_URL}/models/download/${modelName}`);

        if (!response.ok) {
            throw new Error('Download failed');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${modelName}_tb_model.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        hideLoading();
        showAlert('resultsAlert', `Model ${formatModelName(modelName)} berhasil didownload!`, 'success');

    } catch (err) {
        hideLoading();
        showAlert('resultsAlert', 'Error download: ' + err.message, 'error');
    }
}

function renderModelCards(results) {
    const container = document.getElementById('modelCards');
    container.innerHTML = '';
    
    // Sort by AUROC
    const sortedModels = Object.entries(results).sort((a, b) => 
        b[1].metrics.auroc - a[1].metrics.auroc
    );
    
    sortedModels.forEach(([name, data], index) => {
        const m = data.metrics;
        const isBest = index === 0;
        
        const card = document.createElement('div');
        card.className = `model-result-card ${isBest ? 'best' : ''}`;
        
        card.innerHTML = `
            <div class="model-result-header">
                <div class="model-name">${formatModelName(name)}</div>
                ${isBest ? '<div class="best-badge">BEST MODEL</div>' : ''}
            </div>
            <div class="model-metrics">
                <div class="model-metric">
                    <div class="model-metric-value">${(m.auroc * 100).toFixed(1)}%</div>
                    <div class="model-metric-label">AUROC</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-value">${(m.sensitivity * 100).toFixed(1)}%</div>
                    <div class="model-metric-label">Sensitivity</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-value">${(m.specificity * 100).toFixed(1)}%</div>
                    <div class="model-metric-label">Specificity</div>
                </div>
            </div>
            <button class="btn btn-secondary" style="padding: 8px 20px; font-size: 0.85rem;" 
                    onclick="saveModel('${name}')">
                💾 Simpan Model Ini
            </button>
        `;
        
        container.appendChild(card);
    });
}

async function saveModel(modelName) {
    showLoading('Menyimpan model...');
    
    try {
        const response = await fetch(`${API_URL}/models/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                description: `Model ${modelName} dari training batch`,
                tags: ['trained', 'v3']
            })
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (result.success) {
            alert(`Model ${modelName} berhasil disimpan!\n\nPath: ${result.model_path}`);
        } else {
            alert('Gagal menyimpan model: ' + (result.message || 'Unknown error'));
        }
        
    } catch (err) {
        hideLoading();
        alert('Error: ' + err.message);
    }
}

async function saveBestModel() {
    const response = await fetch(`${API_URL}/training/results`);
    const data = await response.json();
    
    if (data.models.length > 0) {
        // Get best model (first in sorted list)
        const bestModel = Object.entries(data.results).sort((a, b) => 
            b[1].metrics.auroc - a[1].metrics.auroc
        )[0][0];
        
        saveModel(bestModel);
    }
}

// ========== MODEL UPLOAD & SELECTION ==========

async function refreshModelList() {
    try {
        const response = await fetch(`${API_URL}/models/list`);
        const data = await response.json();

        const select = document.getElementById('modelSelect');
        select.innerHTML = '<option value="">-- Pilih Model --</option>';

        if (data.success && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                const auroc = model.metrics?.auroc ? (model.metrics.auroc * 100).toFixed(1) + '%' : 'N/A';
                option.textContent = `${formatModelName(model.name)} (AUROC: ${auroc})${model.is_best ? ' ⭐' : ''}`;
                select.appendChild(option);
            });

            // Auto-select best model or first model
            const bestModel = data.models.find(m => m.is_best);
            if (bestModel) {
                select.value = bestModel.name;
                await loadSelectedModel(bestModel.name);
            } else if (data.models.length > 0) {
                select.value = data.models[0].name;
                await loadSelectedModel(data.models[0].name);
            }
        } else {
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "Tidak ada model tersedia - Upload model baru";
            select.appendChild(option);
        }
    } catch (err) {
        console.error('Error refreshing model list:', err);
        showAlert('predictAlert', 'Gagal memuat daftar model', 'error');
    }
}

async function loadSelectedModel(modelName) {
    if (!modelName) return;

    try {
        const formData = new FormData();
        formData.append('model_name', modelName);

        const response = await fetch(`${API_URL}/models/load`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            const info = result.model_info;
            document.getElementById('currentModelInfo').style.display = 'block';
            document.getElementById('activeModelName').textContent = formatModelName(info.name);
            document.getElementById('activeModelBackbone').textContent = info.backbone;
            document.getElementById('activeModelAuroc').textContent = info.metrics?.auroc ?
                (info.metrics.auroc * 100).toFixed(1) + '%' : 'N/A';
        }
    } catch (err) {
        console.error('Error loading model:', err);
    }
}

// Listen for model selection change
document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                loadSelectedModel(e.target.value);
            }
        });
    }
});

async function handleModelUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.zip')) {
        showAlert('predictAlert', 'File harus berformat ZIP', 'error');
        return;
    }

    document.getElementById('uploadedModelName').textContent = file.name;

    showLoading('Mengupload dan mengimport model...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/models/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        hideLoading();

        if (result.success) {
            showAlert('predictAlert', `Model ${result.model_name} berhasil diupload!`, 'success');
            // Refresh model list and select the new model
            await refreshModelList();

            // Select the newly uploaded model
            const select = document.getElementById('modelSelect');
            select.value = result.model_name;
            await loadSelectedModel(result.model_name);

            document.getElementById('uploadedModelName').textContent = '';
            document.getElementById('modelUpload').value = '';
        } else {
            showAlert('predictAlert', 'Upload gagal: ' + (result.message || 'Unknown error'), 'error');
        }

    } catch (err) {
        hideLoading();
        showAlert('predictAlert', 'Error upload: ' + err.message, 'error');
    }
}

// ========== PREDICTION ==========

function handlePredictionAudio(event) {
    const file = event.target.files[0];
    if (!file) return;

    predictionAudio = file;
    document.getElementById('predAudioName').textContent = file.name;
}

async function runPrediction() {
    if (!predictionAudio) {
        showAlert('predictAlert', 'Upload audio batuk terlebih dahulu!', 'warning');
        return;
    }

    // Check if a model is selected
    const selectedModel = document.getElementById('modelSelect').value;
    if (!selectedModel) {
        showAlert('predictAlert', 'Pilih atau upload model terlebih dahulu!', 'warning');
        return;
    }

    showLoading('Menganalisis...');

    try {
        const formData = new FormData();
        formData.append('audio', predictionAudio);
        formData.append('age', document.getElementById('predAge').value);
        formData.append('gender', document.getElementById('predGender').value);
        formData.append('cough_duration_days', document.getElementById('predCoughDuration').value);
        formData.append('has_fever', document.getElementById('predFever').checked);
        formData.append('has_cough', document.getElementById('predCough').checked);
        formData.append('has_night_sweats', document.getElementById('predSweats').checked);
        formData.append('has_weight_loss', document.getElementById('predWeightLoss').checked);
        formData.append('has_chest_pain', document.getElementById('predChestPain').checked);
        formData.append('has_shortness_breath', document.getElementById('predShortness').checked);
        formData.append('previous_tb_history', document.getElementById('predPrevTB').checked);
        formData.append('model_name', selectedModel);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        hideLoading();
        
        const resultDiv = document.getElementById('predictionResult');
        
        if (result.success) {
            const prob = result.result.tb_probability;
            const percentage = Math.round(prob * 100);
            const risk = result.result.risk_level;
            
            let color, emoji, bgColor;
            if (risk === 'RENDAH') {
                color = '#4CAF50';
                emoji = '✅';
                bgColor = '#e8f5e9';
            } else if (risk === 'MENENGAH') {
                color = '#FF9800';
                emoji = '⚠️';
                bgColor = '#fff3e0';
            } else {
                color = '#f44336';
                emoji = '🚨';
                bgColor = '#ffebee';
            }
            
            resultDiv.innerHTML = `
                <div style="font-size: 4rem; margin-bottom: 15px;">${emoji}</div>
                <div style="font-size: 3rem; font-weight: bold; color: ${color}; margin-bottom: 10px;">
                    ${percentage}%
                </div>
                <div style="font-size: 1.2rem; color: ${color}; font-weight: 600; margin-bottom: 15px;">
                    RISIKO ${risk}
                </div>
                <div style="background: ${bgColor}; padding: 15px; border-radius: 10px; text-align: left;">
                    <strong>Model:</strong> ${formatModelName(result.result.model_used)}<br>
                    <strong>Backbone:</strong> ${result.result.backbone_used}<br>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                    ${result.result.recommendation}
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div style="font-size: 4rem; margin-bottom: 15px;">❌</div>
                <div style="color: #c62828;">${result.error || 'Prediksi gagal'}</div>
            `;
        }
        
    } catch (err) {
        hideLoading();
        showAlert('predictAlert', 'Error: ' + err.message, 'error');
    }
}

// ========== STATUS POLLING ==========

async function checkPipelineStatus() {
    try {
        const response = await fetch(`${API_URL}/pipeline/status`);
        const status = await response.json();
        
        datasetUploaded = status.dataset_uploaded;
        preprocessed = status.preprocessed;
        trainingInProgress = status.training_in_progress;
        
        // Update UI based on status
        if (datasetUploaded) {
            document.getElementById('statAudioCount').textContent = status.preprocessed_samples || '-';
            document.getElementById('statStatus').textContent = preprocessed ? 'Preprocessed' : 'Uploaded';
        }
        
        // Update progress if in training
        if (currentSection === 'train' && trainingInProgress) {
            updateProgressUI('train', status.progress, status.current_task);
        } else if (currentSection === 'preprocess' && !preprocessed && status.progress > 0) {
            updateProgressUI('preprocess', status.progress, status.current_task);
        }
        
    } catch (err) {
        console.error('Status check error:', err);
    }
}

function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    statusPollInterval = setInterval(checkPipelineStatus, 2000);
}

function startLogsPolling(type) {
    if (logsPollInterval) clearInterval(logsPollInterval);
    
    logsPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/pipeline/logs?limit=50`);
            const data = await response.json();
            
            const logsContainer = document.getElementById(`${type}Logs`);
            if (logsContainer) {
                logsContainer.innerHTML = data.logs.map(log => {
                    let logClass = 'log-entry';
                    if (log.includes('Error') || log.includes('failed')) logClass += ' error';
                    else if (log.includes('success') || log.includes('complete')) logClass += ' success';
                    else if (log.includes('warning')) logClass += ' warning';
                    return `<div class="${logClass}">${escapeHtml(log)}</div>`;
                }).join('');
                
                // Auto scroll to bottom
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
            
        } catch (err) {
            console.error('Logs polling error:', err);
        }
    }, 1000);
}

function updateProgressUI(type, progress, task) {
    const progressBar = document.getElementById(`${type}Progress`);
    const progressBarOrTask = document.getElementById(`${type}ProgressBar`) || progressBar;
    const taskText = document.getElementById(`${type}Task`);
    
    if (progressBarOrTask) {
        progressBarOrTask.style.width = `${progress}%`;
        progressBarOrTask.textContent = `${progress}%`;
    }
    
    if (taskText && task) {
        taskText.textContent = task;
    }
}

// ========== UTILITY FUNCTIONS ==========

function formatModelName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function showAlert(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `alert alert-${type}`;
    el.classList.remove('hidden');
    
    setTimeout(() => {
        el.classList.add('hidden');
    }, 5000);
}

function showLoading(text) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (statusPollInterval) clearInterval(statusPollInterval);
    if (logsPollInterval) clearInterval(logsPollInterval);
});
