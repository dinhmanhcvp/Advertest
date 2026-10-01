document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('image-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const imgOriginal = document.getElementById('img-original');
    const phOriginal = document.getElementById('ph-original');
    
    const btnSimulate = document.getElementById('btn-simulate');
    const btnLoader = document.getElementById('btn-loader');
    const errMsg = document.getElementById('error-msg');
    
    const severityInput = document.getElementById('severity');
    const sevValDisplay = document.getElementById('sev-val');
    
    const imgSimulated = document.getElementById('img-simulated');
    const phSimulated = document.getElementById('ph-simulated');
    const simStatus = document.getElementById('sim-status');
    
    let currentFile = null;

    // Update Severity Label
    severityInput.addEventListener('input', (e) => {
        sevValDisplay.textContent = e.target.value;
    });

    // Handle Image Upload
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        currentFile = file;
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.add('text-blue-400');
        
        // Display Original
        const reader = new FileReader();
        reader.onload = (e) => {
            imgOriginal.src = e.target.result;
            imgOriginal.classList.remove('hidden');
            phOriginal.classList.add('hidden');
            
            // Reset simulated
            imgSimulated.classList.add('hidden');
            phSimulated.classList.remove('hidden');
            simStatus.textContent = 'Waiting';
            simStatus.className = 'text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-slate-700';
            errMsg.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    });

    // Handle Simulate
    btnSimulate.addEventListener('click', async () => {
        if (!currentFile) {
            showError("Please upload an image first.");
            return;
        }

        // Get selected tags
        const tagCheckboxes = document.querySelectorAll('.attack-tag:checked');
        const tags = Array.from(tagCheckboxes).map(cb => cb.value);
        
        if (tags.length === 0) {
            showError("Please select at least one target error.");
            return;
        }

        const severity = severityInput.value;

        // Prepare FormData
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('tags', JSON.stringify(tags));
        formData.append('severity', severity);

        // UI State: Loading
        setLoading(true);

        try {
            const response = await fetch('http://localhost:8000/api/v1/simulate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Server error: " + response.statusText);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            // Display Result
            imgSimulated.src = url;
            imgSimulated.classList.remove('hidden');
            phSimulated.classList.add('hidden');
            
            simStatus.textContent = 'Success';
            simStatus.className = 'text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30';
            
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnLoader.style.display = 'block';
            btnSimulate.querySelector('span').textContent = 'Processing...';
            btnSimulate.disabled = true;
            btnSimulate.classList.add('opacity-80', 'cursor-not-allowed');
            errMsg.classList.add('hidden');
            simStatus.textContent = 'Processing...';
            simStatus.className = 'text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded border border-blue-500/30';
        } else {
            btnLoader.style.display = 'none';
            btnSimulate.querySelector('span').textContent = 'Run Targeted Simulation';
            btnSimulate.disabled = false;
            btnSimulate.classList.remove('opacity-80', 'cursor-not-allowed');
        }
    }

    }

    // ==========================================
    // Phase 4: End-to-End Pipeline Integration
    // ==========================================
    const API_BASE_URL = 'http://localhost:8000/api/v1';

    // 1. Fetch Insights on load
    async function loadInsights() {
        const container = document.getElementById('insight-container');
        try {
            const res = await fetch(`${API_BASE_URL}/insights`);
            if (!res.ok) throw new Error('Failed to fetch insights');
            const data = await res.json();
            
            container.innerHTML = `
                <div class="mb-4">
                    <span class="text-xs text-slate-400 block mb-1">Total Hard Negatives</span>
                    <strong class="text-2xl text-white">${data.total_hard_negatives.toLocaleString()}</strong>
                </div>
            `;
            
            data.error_distribution.forEach(err => {
                const color = err.tag === 'error_fisheye' ? 'bg-blue-500' : 
                             err.tag === 'error_blur' ? 'bg-purple-500' : 
                             err.tag === 'error_overexposure' ? 'bg-amber-500' : 'bg-slate-500';
                
                container.innerHTML += `
                    <div class="flex flex-col gap-1">
                        <div class="flex justify-between text-xs">
                            <span class="text-slate-300 font-medium">${err.tag}</span>
                            <span class="text-slate-400">${err.percentage}%</span>
                        </div>
                        <div class="w-full bg-slate-800 rounded-full h-1.5">
                            <div class="${color} h-1.5 rounded-full" style="width: ${err.percentage}%"></div>
                        </div>
                    </div>
                `;
            });
        } catch (err) {
            container.innerHTML = `<p class="text-red-400 text-xs">Failed to load insights: ${err.message}</p>`;
        }
    }

    // 2. Fetch Metrics on load
    async function loadMetrics() {
        const container = document.getElementById('metrics-container');
        try {
            const res = await fetch(`${API_BASE_URL}/eval-report`);
            if (!res.ok) throw new Error('Failed to fetch metrics');
            const data = await res.json();
            
            const dropColor = data.deltas.base_drop < -0.05 ? 'text-red-400' : 'text-emerald-400';
            
            container.innerHTML = `
                <div class="grid grid-cols-2 gap-4 h-full">
                    <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 flex flex-col justify-center">
                        <span class="text-[10px] uppercase tracking-wider text-slate-400 mb-2">Base Model</span>
                        <div class="flex justify-between items-baseline border-b border-slate-700 pb-2 mb-2">
                            <span class="text-xs text-slate-300">Clean mAP</span>
                            <span class="text-sm font-bold text-white">${(data.base_model.base_map * 100).toFixed(1)}%</span>
                        </div>
                        <div class="flex justify-between items-baseline">
                            <span class="text-xs text-slate-300">Corrupted mAP</span>
                            <span class="text-sm font-bold text-red-400">${(data.base_model.corrupted_map * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    
                    <div class="bg-blue-900/20 p-4 rounded-xl border border-blue-500/20 flex flex-col justify-center relative overflow-hidden">
                        <div class="absolute -right-4 -top-4 w-16 h-16 bg-blue-500/10 rounded-full blur-xl"></div>
                        <span class="text-[10px] uppercase tracking-wider text-cyan-400 mb-2 font-bold flex items-center gap-1"><i class="fa-solid fa-shield-halved"></i> AdverTest</span>
                        <div class="flex justify-between items-baseline border-b border-blue-500/20 pb-2 mb-2">
                            <span class="text-xs text-slate-300">Clean mAP</span>
                            <span class="text-sm font-bold text-white flex items-center gap-1">${(data.advertest_model.base_map * 100).toFixed(1)}% <span class="text-[9px] ${dropColor}">(${data.deltas.base_drop > 0 ? '+' : ''}${(data.deltas.base_drop * 100).toFixed(1)}%)</span></span>
                        </div>
                        <div class="flex justify-between items-baseline">
                            <span class="text-xs text-slate-300">Corrupted mAP</span>
                            <span class="text-sm font-bold text-emerald-400 flex items-center gap-1">${(data.advertest_model.corrupted_map * 100).toFixed(1)}% <span class="text-[9px] text-emerald-500">(+${(data.deltas.corrupted_improvement * 100).toFixed(1)}%)</span></span>
                        </div>
                    </div>
                </div>
            `;
        } catch (err) {
            container.innerHTML = `<p class="text-red-400 text-xs text-center">Failed to load metrics: ${err.message}</p>`;
        }
    }

    // 3. Handle Pipeline Execution
    const btnRunPipeline = document.getElementById('btn-run-pipeline');
    const jobStatusContainer = document.getElementById('job-status-container');
    const jobStatusText = document.getElementById('job-status-text');
    const jobProgressBar = document.getElementById('job-progress-bar');
    const datasetPathInput = document.getElementById('dataset-path');

    let currentPollInterval = null;

    btnRunPipeline.addEventListener('click', async () => {
        const datasetPath = datasetPathInput.value.trim();
        if (!datasetPath) {
            alert('Please enter a dataset path.');
            return;
        }

        // UI State
        btnRunPipeline.disabled = true;
        btnRunPipeline.classList.add('opacity-50', 'cursor-not-allowed');
        btnRunPipeline.querySelector('span').textContent = 'Starting Pipeline...';
        jobStatusContainer.classList.remove('hidden');
        jobStatusText.textContent = 'Initializing...';
        jobStatusText.className = 'text-[10px] bg-slate-500/20 text-slate-400 px-2 py-0.5 rounded border border-slate-500/30';
        jobProgressBar.style.width = '0%';
        jobProgressBar.className = 'bg-gradient-to-r from-slate-500 to-slate-400 h-1.5 rounded-full transition-all duration-500';

        try {
            const res = await fetch(`${API_BASE_URL}/jobs/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_path: datasetPath })
            });
            if (!res.ok) throw new Error('Failed to start pipeline');
            const data = await res.json();
            
            // Start polling
            pollJobStatus(data.job_id);
            
        } catch (err) {
            jobStatusText.textContent = 'Failed';
            jobStatusText.className = 'text-[10px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/30';
            btnRunPipeline.disabled = false;
            btnRunPipeline.classList.remove('opacity-50', 'cursor-not-allowed');
            btnRunPipeline.querySelector('span').textContent = 'Run AdverTest Pipeline';
            alert(`Error: ${err.message}`);
        }
    });

    async function pollJobStatus(jobId) {
        if (currentPollInterval) clearInterval(currentPollInterval);
        
        currentPollInterval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/jobs/status/${jobId}`);
                if (!res.ok) throw new Error('Polling failed');
                const data = await res.json();
                
                jobStatusText.textContent = data.status;
                jobProgressBar.style.width = `${data.progress}%`;
                
                if (data.status.includes('Training') || data.status.includes('Generating')) {
                    jobStatusText.className = 'text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded border border-blue-500/30';
                    jobProgressBar.className = 'bg-gradient-to-r from-blue-500 to-cyan-400 h-1.5 rounded-full transition-all duration-500';
                }
                
                if (data.status === 'Completed' || data.progress === 100) {
                    clearInterval(currentPollInterval);
                    jobStatusText.className = 'text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30';
                    jobProgressBar.className = 'bg-gradient-to-r from-emerald-500 to-teal-400 h-1.5 rounded-full transition-all duration-500';
                    
                    btnRunPipeline.disabled = false;
                    btnRunPipeline.classList.remove('opacity-50', 'cursor-not-allowed');
                    btnRunPipeline.querySelector('span').textContent = 'Pipeline Completed';
                    
                    // Refresh Metrics after completion
                    loadMetrics();
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 1500);
    }

    // Initialize API-driven components
    setTimeout(() => {
        loadInsights();
        loadMetrics();
    }, 500); // Slight delay for UI rendering

});
