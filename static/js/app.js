/**
 * Clarity AI Studio - Frontend Application Logic
 * Supports Real-ESRGAN and GFPGAN AI Image Enhancement
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropPrompt = document.getElementById('dropPrompt');
    const uploadPreviewWrapper = document.getElementById('uploadPreviewWrapper');
    const uploadThumb = document.getElementById('uploadThumb');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileDims = document.getElementById('previewFileDims');
    const btnRemoveFile = document.getElementById('btnRemoveFile');

    // Batch Enhance Elements
    const enhanceFileCountBadge = document.getElementById('enhanceFileCountBadge');
    const enhanceFilesListWrapper = document.getElementById('enhanceFilesListWrapper');
    const enhanceSelectedCount = document.getElementById('enhanceSelectedCount');
    const btnClearEnhanceFiles = document.getElementById('btnClearEnhanceFiles');
    const enhanceFilesGrid = document.getElementById('enhanceFilesGrid');
    const btnAddMoreEnhanceFiles = document.getElementById('btnAddMoreEnhanceFiles');

    // Batch Enhance Result Elements
    const batchEnhanceProcessingState = document.getElementById('batchEnhanceProcessingState');
    const batchEnhanceProgressBar = document.getElementById('batchEnhanceProgressBar');
    const batchEnhanceProgressPercent = document.getElementById('batchEnhanceProgressPercent');
    const batchEnhanceResultContainer = document.getElementById('batchEnhanceResultContainer');
    const batchEnhanceZipTitle = document.getElementById('batchEnhanceZipTitle');
    const batchEnhanceResultCount = document.getElementById('batchEnhanceResultCount');
    const batchEnhanceResultModel = document.getElementById('batchEnhanceResultModel');
    const batchEnhanceResultTime = document.getElementById('batchEnhanceResultTime');
    const btnSaveBatchEnhanceZipToDownloads = document.getElementById('btnSaveBatchEnhanceZipToDownloads');
    const btnDownloadBatchEnhanceZip = document.getElementById('btnDownloadBatchEnhanceZip');
    const btnResetBatchEnhance = document.getElementById('btnResetBatchEnhance');
    const batchEnhanceResultCountText = document.getElementById('batchEnhanceResultCountText');
    const batchEnhanceResultsGrid = document.getElementById('batchEnhanceResultsGrid');

    const sampleBtns = document.querySelectorAll('.btn-sample');
    const modelCards = document.querySelectorAll('.model-card');
    const scaleBtns = document.querySelectorAll('.btn-scale');
    const faceEnhanceToggle = document.getElementById('faceEnhanceToggle');
    const btnEnhance = document.getElementById('btnEnhance');
    const btnEnhanceText = document.getElementById('btnEnhanceText');

    // Workspace States
    const emptyState = document.getElementById('emptyState');
    const processingState = document.getElementById('processingState');
    const resultContainer = document.getElementById('resultContainer');
    const processingStatusText = document.getElementById('processingStatusText');
    const processingSubText = document.getElementById('processingSubText');

    // Comparison Elements
    const comparisonStage = document.getElementById('comparisonStage');
    const sliderWrapper = document.getElementById('sliderWrapper');
    const sideWrapper = document.getElementById('sideWrapper');
    const imgEnhanced = document.getElementById('imgEnhanced');
    const imgOriginal = document.getElementById('imgOriginal');
    const sliderClip = document.getElementById('sliderClip');
    const sliderHandle = document.getElementById('sliderHandle');
    const sideImgOrig = document.getElementById('sideImgOrig');
    const sideImgEnh = document.getElementById('sideImgEnh');

    // View tabs & Zoom
    const viewTabs = document.querySelectorAll('.view-tab');
    const btnZoomFit = document.getElementById('btnZoomFit');
    const btnZoomToggle = document.getElementById('btnZoomToggle');
    const btnReset = document.getElementById('btnReset');
    const btnDownload = document.getElementById('btnDownload');
    const btnDownloadJpg = document.getElementById('btnDownloadJpg');
    const btnSaveToDownloads = document.getElementById('btnSaveToDownloads');

    // Stats Bar
    const statResolution = document.getElementById('statResolution');
    const statPixels = document.getElementById('statPixels');
    const statFilesize = document.getElementById('statFilesize');
    const statDuration = document.getElementById('statDuration');

    // History
    const historySection = document.getElementById('historySection');
    const historyList = document.getElementById('historyList');

    // State Variables
    let currentFile = null;
    let currentSampleId = null;
    let selectedEnhanceFiles = [];
    let selectedScale = 4;
    let isProcessing = false;
    let currentResultData = null;
    let isZoomed = false;

    // Slider Dragging State
    let isDraggingSlider = false;
    let sliderPos = 50; // percentage

    // Fetch System Info on Load
    fetchSystemInfo();
    fetchHistory();

    // 1. File Upload & Drag-and-Drop
    dropZone.addEventListener('click', (e) => {
        if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleEnhanceFilesSelect(e.target.files);
        }
    });

    if (btnAddMoreEnhanceFiles) {
        btnAddMoreEnhanceFiles.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (btnClearEnhanceFiles) {
        btnClearEnhanceFiles.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedEnhanceFiles = [];
            resetFileInput();
        });
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleEnhanceFilesSelect(dt.files);
        }
    });

    // Clipboard Paste (Ctrl+V)
    window.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                handleEnhanceFilesSelect([blob], 'Ảnh từ Clipboard.png');
                showToast('Đã dán ảnh từ Clipboard thành công!');
                break;
            }
        }
    });

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileInput();
    });

    function handleEnhanceFilesSelect(files, customName = null) {
        const fileArr = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (fileArr.length === 0) {
            showToast('Vui lòng chọn tệp hình ảnh hợp lệ (PNG, JPG, WEBP, BMP)', 'error');
            return;
        }

        currentSampleId = null;

        // Auto recommend Anime model if PNG or cartoon-like name
        const hasAnimeHint = fileArr.some(f => f.name.toLowerCase().endsWith('.png') || f.name.toLowerCase().includes('anime') || f.name.toLowerCase().includes('art'));
        if (hasAnimeHint) {
            const animeRadio = document.getElementById('modelAnime');
            if (animeRadio) {
                animeRadio.checked = true;
                updateActiveModelCard();
                showToast('💡 Đã tự động chọn mô hình Anime Cel Art (Tối ưu sắc nét viền vẽ 2D / sticker)!');
            }
        }

        // If currently in single mode and adding more
        if (fileArr.length === 1 && selectedEnhanceFiles.length === 0) {
            currentFile = fileArr[0];
            selectedEnhanceFiles = [currentFile];

            const reader = new FileReader();
            reader.onload = (e) => {
                const dataUrl = e.target.result;
                const img = new Image();
                img.onload = () => {
                    uploadThumb.src = dataUrl;
                    previewFileName.textContent = customName || currentFile.name;
                    const sizeKb = (currentFile.size / 1024).toFixed(1);
                    previewFileDims.textContent = `${img.naturalWidth} x ${img.naturalHeight} px · ${sizeKb} KB`;

                    dropPrompt.classList.add('hidden');
                    uploadPreviewWrapper.classList.remove('hidden');
                    if (enhanceFilesListWrapper) enhanceFilesListWrapper.classList.add('hidden');
                    if (enhanceFileCountBadge) enhanceFileCountBadge.classList.add('hidden');
                    btnEnhance.disabled = false;
                    btnEnhanceText.textContent = 'Bắt Đầu Làm Nét Chi Tiết';
                };
                img.src = dataUrl;
            };
            reader.readAsDataURL(currentFile);
        } else {
            // Multi-image batch mode
            fileArr.forEach(f => {
                // Avoid duplicates by name + size
                if (!selectedEnhanceFiles.some(existing => existing.name === f.name && existing.size === f.size)) {
                    selectedEnhanceFiles.push(f);
                }
            });

            currentFile = null;
            dropPrompt.classList.add('hidden');
            uploadPreviewWrapper.classList.add('hidden');
            if (enhanceFilesListWrapper) enhanceFilesListWrapper.classList.remove('hidden');
            if (enhanceFileCountBadge) {
                enhanceFileCountBadge.classList.remove('hidden');
                enhanceFileCountBadge.textContent = `${selectedEnhanceFiles.length} ảnh`;
            }
            if (enhanceSelectedCount) enhanceSelectedCount.textContent = selectedEnhanceFiles.length;

            renderEnhanceFilesGrid();
            btnEnhance.disabled = false;
            btnEnhanceText.textContent = `⚡ Bắt Đầu Làm Nét Hàng Loạt (${selectedEnhanceFiles.length} ảnh)`;
        }
    }

    function renderEnhanceFilesGrid() {
        if (!enhanceFilesGrid) return;
        enhanceFilesGrid.innerHTML = '';
        selectedEnhanceFiles.forEach((file, idx) => {
            const card = document.createElement('div');
            card.className = 'batch-file-card';
            card.style.cssText = 'display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.05); padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);';
            
            const thumbUrl = URL.createObjectURL(file);
            const sizeKb = (file.size / 1024).toFixed(1);

            card.innerHTML = `
                <img src="${thumbUrl}" style="width: 36px; height: 36px; border-radius: 6px; object-fit: cover;">
                <div style="flex: 1; overflow: hidden; min-width: 0;">
                    <div style="font-size: 12px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${file.name}</div>
                    <div style="font-size: 11px; color: var(--text-muted);">${sizeKb} KB</div>
                </div>
                <button type="button" class="btn-remove-mini" data-idx="${idx}" style="background: transparent; border: none; color: #ef4444; cursor: pointer; padding: 4px;" title="Xóa">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            `;

            card.querySelector('.btn-remove-mini').addEventListener('click', (e) => {
                e.stopPropagation();
                selectedEnhanceFiles.splice(idx, 1);
                if (selectedEnhanceFiles.length === 0) {
                    resetFileInput();
                } else if (selectedEnhanceFiles.length === 1) {
                    handleEnhanceFilesSelect([selectedEnhanceFiles[0]]);
                } else {
                    if (enhanceSelectedCount) enhanceSelectedCount.textContent = selectedEnhanceFiles.length;
                    if (enhanceFileCountBadge) enhanceFileCountBadge.textContent = `${selectedEnhanceFiles.length} ảnh`;
                    btnEnhanceText.textContent = `⚡ Bắt Đầu Làm Nét Hàng Loạt (${selectedEnhanceFiles.length} ảnh)`;
                    renderEnhanceFilesGrid();
                }
            });

            enhanceFilesGrid.appendChild(card);
        });
    }

    function resetFileInput() {
        currentFile = null;
        currentSampleId = null;
        selectedEnhanceFiles = [];
        fileInput.value = '';
        dropPrompt.classList.remove('hidden');
        uploadPreviewWrapper.classList.add('hidden');
        if (enhanceFilesListWrapper) enhanceFilesListWrapper.classList.add('hidden');
        if (enhanceFileCountBadge) enhanceFileCountBadge.classList.add('hidden');
        btnEnhance.disabled = true;
        btnEnhanceText.textContent = 'Bắt Đầu Làm Nét Chi Tiết';
    }

    // 2. Sample Presets
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleType = btn.dataset.sample;
            loadSample(sampleType);
        });
    });

    function loadSample(sampleType) {
        currentSampleId = sampleType;
        currentFile = null;

        if (sampleType === 'portrait') {
            uploadThumb.src = '/static/samples/sample_portrait_lowres.jpg';
            previewFileName.textContent = 'sample_portrait_lowres.jpg (Chân dung cũ)';
            previewFileDims.textContent = '280 x 280 px · Ảnh mẫu nén';
            
            // Auto configure best model
            document.getElementById('modelCombo').checked = true;
            updateActiveModelCard();
            selectedScale = 4;
            updateScaleButtons();
            faceEnhanceToggle.checked = true;

        } else if (sampleType === 'anime') {
            uploadThumb.src = '/static/samples/sample_anime_lowres.jpg';
            previewFileName.textContent = 'sample_anime_lowres.jpg (Cyberpunk Anime)';
            previewFileDims.textContent = '280 x 280 px · Ảnh mẫu nén';
            
            // Auto configure best model
            document.getElementById('modelAnime').checked = true;
            updateActiveModelCard();
            selectedScale = 4;
            updateScaleButtons();
            faceEnhanceToggle.checked = false;
        }

        dropPrompt.classList.add('hidden');
        uploadPreviewWrapper.classList.remove('hidden');
        btnEnhance.disabled = false;
        showToast(`Đã nạp ảnh mẫu: ${sampleType === 'portrait' ? 'Chân dung cổ điển' : 'Anime'}!`);
    }

    // 3. Model & Parameter Configuration
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            setTimeout(updateActiveModelCard, 50);
        });
    });

    function updateActiveModelCard() {
        modelCards.forEach(card => {
            const input = card.querySelector('input');
            if (input.checked) {
                card.classList.add('active');
                if (input.value === 'scunet') {
                    selectedScale = 1;
                    updateScaleButtons();
                }
            } else {
                card.classList.remove('active');
            }
        });
    }

    scaleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            selectedScale = parseInt(btn.dataset.scale);
            updateScaleButtons();
        });
    });

    function updateScaleButtons() {
        scaleBtns.forEach(b => {
            if (parseInt(b.dataset.scale) === selectedScale) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
    }

    // 4. Enhance Action
    btnEnhance.addEventListener('click', () => {
        if (isProcessing) return;
        if (selectedEnhanceFiles && selectedEnhanceFiles.length > 1) {
            runBatchEnhance();
        } else {
            runEnhance();
        }
    });

    async function runEnhance() {
        if (!currentFile && !currentSampleId) {
            showToast('Vui lòng chọn hoặc tải lên một hình ảnh trước!', 'error');
            return;
        }

        isProcessing = true;
        btnEnhance.disabled = true;
        btnEnhanceText.textContent = 'Đang xử lý trên GPU...';

        // Update UI states
        emptyState.classList.add('hidden');
        resultContainer.classList.add('hidden');
        if (batchEnhanceResultContainer) batchEnhanceResultContainer.classList.add('hidden');
        processingState.classList.remove('hidden');

        // Dynamic step animation
        animateProcessingSteps();

        const selectedModel = document.querySelector('input[name="modelType"]:checked')?.value || 'combo';
        const isFaceEnhance = faceEnhanceToggle.checked;

        const formData = new FormData();
        formData.append('model_type', selectedModel);
        formData.append('outscale', selectedScale);
        formData.append('face_enhance', isFaceEnhance ? 'true' : 'false');

        if (currentSampleId) {
            formData.append('sample_id', currentSampleId);
        } else if (currentFile) {
            formData.append('image', currentFile);
        }

        try {
            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: formData
            });

            const res = await response.json();

            if (res.status === 'success') {
                displayResult(res.data);
                showToast(`Làm nét hoàn tất trong ${res.data.stats.duration_sec}s!`);
                fetchHistory();
            } else {
                throw new Error(res.message || 'Lỗi xử lý ảnh.');
            }
        } catch (err) {
            console.error('Enhance error:', err);
            showToast(`Lỗi: ${err.message}`, 'error');
            processingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            isProcessing = false;
            btnEnhance.disabled = false;
            btnEnhanceText.textContent = 'Bắt Đầu Làm Nét Chi Tiết';
        }
    }

    // Batch Enhance Flow (Nhiều ảnh cùng lúc)
    async function runBatchEnhance() {
        if (!selectedEnhanceFiles || selectedEnhanceFiles.length === 0) return;

        isProcessing = true;
        btnEnhance.disabled = true;
        btnEnhanceText.textContent = `Đang xử lý ${selectedEnhanceFiles.length} ảnh trên GPU...`;

        emptyState.classList.add('hidden');
        resultContainer.classList.add('hidden');
        if (batchEnhanceResultContainer) batchEnhanceResultContainer.classList.add('hidden');
        if (batchEnhanceProcessingState) batchEnhanceProcessingState.classList.remove('hidden');

        const selectedModel = document.querySelector('input[name="modelType"]:checked')?.value || 'anime';
        const isFaceEnhance = faceEnhanceToggle.checked;

        const formData = new FormData();
        formData.append('model_type', selectedModel);
        formData.append('outscale', selectedScale);
        formData.append('face_enhance', isFaceEnhance ? 'true' : 'false');

        selectedEnhanceFiles.forEach(f => {
            formData.append('images', f);
        });

        // Dynamic progress bar
        let progress = 10;
        if (batchEnhanceProgressBar) batchEnhanceProgressBar.style.width = '10%';
        if (batchEnhanceProgressPercent) batchEnhanceProgressPercent.textContent = `Đang tải ${selectedEnhanceFiles.length} ảnh vào VRAM (10%)...`;

        const progressTimer = setInterval(() => {
            if (!isProcessing) {
                clearInterval(progressTimer);
                return;
            }
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 4;
                if (progress > 90) progress = 90;
                if (batchEnhanceProgressBar) batchEnhanceProgressBar.style.width = `${progress}%`;
                const currentItem = Math.min(selectedEnhanceFiles.length, Math.ceil((progress / 100) * selectedEnhanceFiles.length));
                if (batchEnhanceProgressPercent) {
                    batchEnhanceProgressPercent.textContent = `Đang xử lý ảnh ${currentItem}/${selectedEnhanceFiles.length} trên RTX 3060 (${progress}%)...`;
                }
            }
        }, 600);

        try {
            const response = await fetch('/api/batch-enhance', {
                method: 'POST',
                body: formData
            });

            const res = await response.json();
            clearInterval(progressTimer);

            if (res.status === 'success') {
                if (batchEnhanceProgressBar) batchEnhanceProgressBar.style.width = '100%';
                if (batchEnhanceProgressPercent) batchEnhanceProgressPercent.textContent = 'Hoàn tất 100%!';
                displayBatchEnhanceResult(res.data);
                showToast(`Đã làm nét xong ${res.data.total_files} ảnh trong ${res.data.total_duration_sec}s!`);
                fetchHistory();
            } else {
                throw new Error(res.message || 'Lỗi khi làm nét hàng loạt.');
            }
        } catch (err) {
            console.error('Batch enhance error:', err);
            showToast(`Lỗi: ${err.message}`, 'error');
            if (batchEnhanceProcessingState) batchEnhanceProcessingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            isProcessing = false;
            btnEnhance.disabled = false;
            btnEnhanceText.textContent = `⚡ Bắt Đầu Làm Nét Hàng Loạt (${selectedEnhanceFiles.length} ảnh)`;
        }
    }

    function displayBatchEnhanceResult(data) {
        if (batchEnhanceProcessingState) batchEnhanceProcessingState.classList.add('hidden');
        if (batchEnhanceResultContainer) batchEnhanceResultContainer.classList.remove('hidden');

        if (batchEnhanceZipTitle) batchEnhanceZipTitle.textContent = data.zip_filename;
        if (batchEnhanceResultCount) batchEnhanceResultCount.textContent = data.total_files;
        if (batchEnhanceResultCountText) batchEnhanceResultCountText.textContent = data.total_files;
        if (batchEnhanceResultModel) batchEnhanceResultModel.textContent = data.model_used === 'anime' ? 'ANIME (CEL ART)' : data.model_used.toUpperCase();
        if (batchEnhanceResultTime) batchEnhanceResultTime.textContent = `${data.total_duration_sec}s (tb: ${data.avg_duration_sec}s/ảnh)`;

        if (btnDownloadBatchEnhanceZip) {
            btnDownloadBatchEnhanceZip.onclick = () => {
                window.location.href = data.zip_url;
            };
        }

        if (btnSaveBatchEnhanceZipToDownloads) {
            btnSaveBatchEnhanceZipToDownloads.onclick = () => {
                saveBatchEnhanceZipToDownloads(data.zip_filename);
            };
        }

        if (btnResetBatchEnhance) {
            btnResetBatchEnhance.onclick = () => {
                selectedEnhanceFiles = [];
                resetFileInput();
                if (batchEnhanceResultContainer) batchEnhanceResultContainer.classList.add('hidden');
                emptyState.classList.remove('hidden');
            };
        }

        // Render Gallery Grid
        if (batchEnhanceResultsGrid) {
            batchEnhanceResultsGrid.innerHTML = '';
            data.images.forEach(item => {
                const card = document.createElement('div');
                card.className = 'batch-result-card glass-card';
                card.style.cssText = 'padding: 12px; border-radius: 12px; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); display: flex; flex-direction: column; gap: 10px;';
                
                card.innerHTML = `
                    <div style="position: relative; border-radius: 8px; overflow: hidden; height: 180px; background: #0b0f19; display: flex; align-items: center; justify-content: center;">
                        <img src="${item.enhanced_url}" alt="${item.enhanced_filename}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                        <button type="button" class="btn-preview-split" style="position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.75); color: #00f0ff; border: 1px solid rgba(0,240,255,0.4); border-radius: 6px; padding: 4px 8px; font-size: 11px; cursor: pointer; display: flex; align-items: center; gap: 4px;" title="Xem thanh trượt so sánh ảnh này">
                            <i class="fa-solid fa-arrows-left-right"></i> So sánh
                        </button>
                    </div>
                    <div style="font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${item.original_filename}">
                        ${item.original_filename}
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); display: flex; justify-content: space-between;">
                        <span>${item.stats.orig_width}x${item.stats.orig_height} ➔ <strong style="color: #00f0ff;">${item.stats.final_width}x${item.stats.final_height}</strong></span>
                        <span>${item.orig_size_kb} KB ➔ <strong style="color: #10b981;">${item.enhanced_size_kb} KB</strong></span>
                    </div>
                    <div style="display: flex; gap: 8px; margin-top: 4px;">
                        <a href="${item.download_url}" class="btn-secondary-sm" style="flex: 1; text-align: center; text-decoration: none; padding: 6px 8px; font-size: 12px; border-radius: 6px; background: rgba(255,255,255,0.06); color: white; display: inline-flex; align-items: center; justify-content: center; gap: 4px;">
                            <i class="fa-solid fa-download"></i> Tải PNG
                        </a>
                        <button type="button" class="btn-save-item-downloads" style="flex: 1; padding: 6px 8px; font-size: 12px; border-radius: 6px; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 4px;" title="Lưu thẳng vào Downloads">
                            <i class="fa-solid fa-folder-arrow-down"></i> Downloads
                        </button>
                    </div>
                `;

                // Wire individual download save
                card.querySelector('.btn-save-item-downloads').onclick = () => {
                    saveBatchEnhanceItemToDownloads(item.batch_id, item.enhanced_filename);
                };

                // Wire "So sánh" button to open comparison slider for this image!
                card.querySelector('.btn-preview-split').onclick = () => {
                    if (batchEnhanceResultContainer) batchEnhanceResultContainer.classList.add('hidden');
                    displayResult({
                        enhanced_url: item.enhanced_url,
                        original_url: item.original_url,
                        download_url: item.download_url,
                        download_jpg_url: item.download_url,
                        filename: item.enhanced_filename,
                        original_filename: item.original_filename,
                        stats: item.stats,
                        file_sizes: {
                            original_kb: item.orig_size_kb,
                            enhanced_kb: item.enhanced_size_kb,
                            enhanced_mb: (item.enhanced_size_kb / 1024).toFixed(2)
                        }
                    });
                    showToast(`Đang xem so sánh Trước/Sau: ${item.original_filename}`);
                };

                batchEnhanceResultsGrid.appendChild(card);
            });
        }
    }

    async function saveBatchEnhanceZipToDownloads(filename) {
        try {
            const formData = new FormData();
            formData.append('filename', filename);
            const res = await fetch('/api/save-batch-zip-to-downloads', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast(`Đã lưu file ZIP vào Downloads: ${filename}`);
            } else {
                throw new Error(data.message);
            }
        } catch (e) {
            showToast(`Lỗi: ${e.message}`, 'error');
        }
    }

    async function saveBatchEnhanceItemToDownloads(batchId, filename) {
        try {
            const formData = new FormData();
            formData.append('batch_id', batchId);
            formData.append('filename', filename);
            const res = await fetch('/api/save-batch-enhance-item-to-downloads', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast(`Đã lưu ${filename} vào Downloads!`);
            } else {
                throw new Error(data.message);
            }
        } catch (e) {
            showToast(`Lỗi: ${e.message}`, 'error');
        }
    }

    function animateProcessingSteps() {
        const steps = [
            { text: 'Đang tải ảnh vào VRAM...', sub: 'Khởi tạo bộ đệm ma trận Tensor trên NVIDIA RTX 3060...' },
            { text: 'Chạy mạng nơ-ron Real-ESRGAN...', sub: 'Đang tái cấu trúc các chi tiết bị vỡ và khử nhiễu ảnh...' },
            { text: 'Phục hồi chi tiết & khuôn mặt GFPGAN...', sub: 'Tái tạo đường nét sắc sảo chân thực...' },
            { text: 'Xuất kết quả độ phân giải cao...', sub: 'Hoàn tất ghép mảng pixel và lưu ảnh 4K...' }
        ];

        let i = 0;
        const interval = setInterval(() => {
            if (!isProcessing) {
                clearInterval(interval);
                return;
            }
            if (i < steps.length) {
                processingStatusText.textContent = steps[i].text;
                processingSubText.textContent = steps[i].sub;
                i++;
            }
        }, 1200);
    }

    // 5. Display Result & Slider Setup
    function displayResult(data) {
        currentResultData = data;
        processingState.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        // Set Image Sources
        imgEnhanced.src = data.enhanced_url;
        imgOriginal.src = data.original_url;
        sideImgOrig.src = data.original_url;
        sideImgEnh.src = data.enhanced_url;

        // Update Stats Bar
        const s = data.stats;
        statResolution.textContent = `${s.orig_width}x${s.orig_height} ➔ ${s.final_width}x${s.final_height}`;
        statPixels.textContent = `${s.orig_mp} MP ➔ ${s.final_mp} MP (+${(s.final_mp / Math.max(s.orig_mp, 0.01)).toFixed(0)}x)`;
        statFilesize.textContent = `${data.file_sizes.original_kb} KB ➔ ${data.file_sizes.enhanced_mb} MB`;
        statDuration.textContent = `⚡ ${s.duration_sec} giây`;

        // Reset Slider to 50%
        setSliderPosition(50);
        setViewMode('slider');

        // Ensure clip image sync after load
        imgEnhanced.onload = syncSliderImages;
        window.addEventListener('resize', syncSliderImages);
    }

    function syncSliderImages() {
        if (!imgEnhanced.complete) return;
        const renderedW = imgEnhanced.offsetWidth;
        const renderedH = imgEnhanced.offsetHeight;

        // Sync original image dimensions inside clip
        imgOriginal.style.width = `${renderedW}px`;
        imgOriginal.style.height = `${renderedH}px`;
    }

    // Slider Dragging Logic
    function setSliderPosition(percentage) {
        sliderPos = Math.max(0, Math.min(100, percentage));
        sliderClip.style.width = `${sliderPos}%`;
        sliderHandle.style.left = `${sliderPos}%`;
    }

    function onSliderMove(e) {
        if (!isDraggingSlider) return;
        const rect = sliderWrapper.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const offsetX = clientX - rect.left;
        const percentage = (offsetX / rect.width) * 100;
        setSliderPosition(percentage);
    }

    sliderHandle.addEventListener('mousedown', (e) => {
        isDraggingSlider = true;
        sliderHandle.classList.add('dragging');
        e.preventDefault();
    });

    sliderWrapper.addEventListener('mousedown', (e) => {
        isDraggingSlider = true;
        sliderHandle.classList.add('dragging');
        onSliderMove(e);
    });

    window.addEventListener('mousemove', onSliderMove);
    window.addEventListener('mouseup', () => {
        if (isDraggingSlider) {
            isDraggingSlider = false;
            sliderHandle.classList.remove('dragging');
        }
    });

    // Touch Support
    sliderHandle.addEventListener('touchstart', (e) => {
        isDraggingSlider = true;
        sliderHandle.classList.add('dragging');
    });

    window.addEventListener('touchmove', onSliderMove);
    window.addEventListener('touchend', () => {
        if (isDraggingSlider) {
            isDraggingSlider = false;
            sliderHandle.classList.remove('dragging');
        }
    });

    // 6. View Mode Switcher
    viewTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const mode = tab.dataset.mode;
            setViewMode(mode);
        });
    });

    function setViewMode(mode) {
        viewTabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));

        if (mode === 'slider') {
            sliderWrapper.classList.remove('hidden');
            sideWrapper.classList.add('hidden');
            sliderClip.classList.remove('hidden');
            sliderHandle.classList.remove('hidden');
            setSliderPosition(50);
            setTimeout(syncSliderImages, 50);
        } else if (mode === 'side') {
            sliderWrapper.classList.add('hidden');
            sideWrapper.classList.remove('hidden');
        } else if (mode === 'enhanced') {
            sliderWrapper.classList.remove('hidden');
            sideWrapper.classList.add('hidden');
            sliderClip.classList.add('hidden');
            sliderHandle.classList.add('hidden');
        } else if (mode === 'original') {
            sliderWrapper.classList.remove('hidden');
            sideWrapper.classList.add('hidden');
            sliderClip.classList.remove('hidden');
            sliderClip.style.width = '100%';
            sliderHandle.classList.add('hidden');
            setTimeout(syncSliderImages, 50);
        }
    }

    // 7. Zoom & Reset Controls
    btnZoomToggle.addEventListener('click', () => {
        isZoomed = !isZoomed;
        if (isZoomed) {
            imgEnhanced.style.transform = 'scale(1.8)';
            imgOriginal.style.transform = 'scale(1.8)';
            imgEnhanced.style.transition = 'transform 0.3s ease';
            imgOriginal.style.transition = 'transform 0.3s ease';
            btnZoomToggle.innerHTML = '<i class="fa-solid fa-magnifying-glass-minus"></i>';
            showToast('Đã phóng to chi tiết 180%!');
        } else {
            resetZoom();
        }
    });

    btnZoomFit.addEventListener('click', resetZoom);

    function resetZoom() {
        isZoomed = false;
        imgEnhanced.style.transform = 'none';
        imgOriginal.style.transform = 'none';
        btnZoomToggle.innerHTML = '<i class="fa-solid fa-magnifying-glass-plus"></i>';
    }

    btnReset.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        emptyState.classList.remove('hidden');
        resetFileInput();
    });

    // 7.1 Save directly to Windows Downloads folder
    if (btnSaveToDownloads) {
        btnSaveToDownloads.addEventListener('click', async () => {
            if (!currentResultData) return;
            const filename = currentResultData.filename;
            showToast('Đang lưu tệp vào thư mục Downloads...');
            try {
                const formData = new FormData();
                formData.append('filename', filename);
                const res = await fetch('/api/save-to-downloads', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp vào Downloads: ${data.filename}`);
                } else {
                    throw new Error(data.message || 'Lỗi khi lưu tệp');
                }
            } catch (err) {
                console.error('Save to downloads error:', err);
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    // Download handlers (Ensures proper .png / .jpg extension)
    btnDownload.addEventListener('click', () => {
        if (!currentResultData) return;
        const filename = `ClarityAI_enhanced_${currentResultData.id}.png`;
        const url = currentResultData.download_url || `/api/download/${currentResultData.filename}`;
        triggerDownload(url, filename);
    });

    if (btnDownloadJpg) {
        btnDownloadJpg.addEventListener('click', () => {
            if (!currentResultData) return;
            const filename = `ClarityAI_enhanced_${currentResultData.id}.jpg`;
            const jpgUrl = currentResultData.download_jpg_url || `/api/download-jpg/${currentResultData.filename}`;
            triggerDownload(jpgUrl, filename);
        });
    }

    async function triggerDownload(url, filename) {
        showToast(`Đang tải ảnh: ${filename}...`);
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error('Không thể tải tệp từ máy chủ');
            const blob = await res.blob();
            const blobUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                window.URL.revokeObjectURL(blobUrl);
                document.body.removeChild(a);
            }, 1000);
            showToast(`Đã tải xuống thành công: ${filename}!`);
        } catch (e) {
            console.warn('Blob download failed, using standard link', e);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    }

    // 8. System Info
    async function fetchSystemInfo() {
        try {
            const res = await fetch('/api/system-info');
            const data = await res.json();
            if (data.status === 'success') {
                const info = data.info;
                const gpuText = `${info.gpu_name} · ${info.vram_total_gb}GB CUDA`;
                document.getElementById('gpuStatusText').textContent = gpuText;
            }
        } catch (e) {
            console.warn('System info check failed', e);
        }
    }

    // 9. History Gallery
    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            if (data.status === 'success' && data.history.length > 0) {
                renderHistory(data.history);
            }
        } catch (e) {
            console.warn('History check failed', e);
        }
    }

    function renderHistory(list) {
        historySection.classList.remove('hidden');
        historyList.innerHTML = '';
        list.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.title = `${item.original_filename} (${item.timestamp})`;
            const thumbSrc = item.enhanced_url || item.result_url || item.original_url;
            div.innerHTML = `<img src="${thumbSrc}" alt="Thumbnail">`;
            div.addEventListener('click', () => {
                if (item.result_url && tabBtnAspect) {
                    tabBtnAspect.click();
                    displayAspectResult(item);
                } else {
                    tabBtnEnhancer.click();
                    displayResult(item);
                }
            });
            historyList.appendChild(div);
        });
    }

    // Toast Helper
    let toastTimeout;
    function showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastMsg = document.getElementById('toastMsg');
        const toastIcon = document.getElementById('toastIcon');

        toastMsg.textContent = message;
        if (type === 'error') {
            toast.style.borderColor = '#ef4444';
            toastIcon.className = 'fa-solid fa-circle-exclamation';
            toastIcon.style.color = '#ef4444';
        } else {
            toast.style.borderColor = 'var(--primary)';
            toastIcon.className = 'fa-solid fa-circle-check';
            toastIcon.style.color = '#34d399';
        }

        toast.classList.remove('hidden');
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toast.classList.add('hidden');
        }, 3500);
    }
    window.showToast = showToast;

    // ==========================================
    // 10. TOP NAV TAB SWITCHER (Enhance vs Aspect vs Batch vs Compress vs Generator vs Chatbot vs BgRemover vs Converter)
    // ==========================================
    const tabBtnEnhancer = document.getElementById('tabBtnEnhancer');
    const tabBtnAspect = document.getElementById('tabBtnAspect');
    const tabBtnBatchResize = document.getElementById('tabBtnBatchResize');
    const tabBtnGenerator = document.getElementById('tabBtnGenerator');
    const tabBtnChatbot = document.getElementById('tabBtnChatbot');
    const tabBtnCompressor = document.getElementById('tabBtnCompressor');
    const tabBtnBgRemover = document.getElementById('tabBtnBgRemover');
    const tabBtnConverter = document.getElementById('tabBtnConverter');

    const viewEnhancer = document.getElementById('viewEnhancer');
    const viewAspect = document.getElementById('viewAspect');
    const viewBatchResize = document.getElementById('viewBatchResize');
    const viewGenerator = document.getElementById('viewGenerator');
    const viewChatbot = document.getElementById('viewChatbot');
    const viewCompressor = document.getElementById('viewCompressor');
    const viewBgRemover = document.getElementById('viewBgRemover');
    const viewConverter = document.getElementById('viewConverter');

    function switchAppTab(activeTab) {
        // Tab buttons
        tabBtnEnhancer.classList.toggle('active', activeTab === 'enhancer');
        tabBtnAspect.classList.toggle('active', activeTab === 'aspect');
        if (tabBtnBatchResize) tabBtnBatchResize.classList.toggle('active', activeTab === 'batch');
        if (tabBtnGenerator) tabBtnGenerator.classList.toggle('active', activeTab === 'generator');
        if (tabBtnChatbot) tabBtnChatbot.classList.toggle('active', activeTab === 'chatbot');
        if (tabBtnCompressor) tabBtnCompressor.classList.toggle('active', activeTab === 'compress');
        if (tabBtnBgRemover) tabBtnBgRemover.classList.toggle('active', activeTab === 'bgremover');
        if (tabBtnConverter) tabBtnConverter.classList.toggle('active', activeTab === 'converter');

        // Views
        viewEnhancer.classList.toggle('hidden', activeTab !== 'enhancer');
        viewAspect.classList.toggle('hidden', activeTab !== 'aspect');
        if (viewBatchResize) viewBatchResize.classList.toggle('hidden', activeTab !== 'batch');
        if (viewGenerator) viewGenerator.classList.toggle('hidden', activeTab !== 'generator');
        if (viewChatbot) viewChatbot.classList.toggle('hidden', activeTab !== 'chatbot');
        if (viewCompressor) viewCompressor.classList.toggle('hidden', activeTab !== 'compress');
        if (viewBgRemover) viewBgRemover.classList.toggle('hidden', activeTab !== 'bgremover');
        if (viewConverter) viewConverter.classList.toggle('hidden', activeTab !== 'converter');

        // Auto scroll to chat bottom if opening chatbot
        if (activeTab === 'chatbot') {
            const chatBox = document.getElementById('chatMessagesContainer');
            if (chatBox) setTimeout(() => { chatBox.scrollTop = chatBox.scrollHeight; }, 100);
        }
    }

    if (tabBtnEnhancer) {
        tabBtnEnhancer.addEventListener('click', () => switchAppTab('enhancer'));
    }
    if (tabBtnAspect) {
        tabBtnAspect.addEventListener('click', () => switchAppTab('aspect'));
    }
    if (tabBtnBatchResize) {
        tabBtnBatchResize.addEventListener('click', () => switchAppTab('batch'));
    }
    if (tabBtnGenerator) {
        tabBtnGenerator.addEventListener('click', () => switchAppTab('generator'));
    }
    if (tabBtnChatbot) {
        tabBtnChatbot.addEventListener('click', () => switchAppTab('chatbot'));
    }
    if (tabBtnCompressor) {
        tabBtnCompressor.addEventListener('click', () => switchAppTab('compress'));
    }
    if (tabBtnBgRemover) {
        tabBtnBgRemover.addEventListener('click', () => switchAppTab('bgremover'));
    }
    if (tabBtnConverter) {
        tabBtnConverter.addEventListener('click', () => switchAppTab('converter'));
    }

    // ==========================================
    // 11. ASPECT RATIO STUDIO CLIENT LOGIC
    // ==========================================
    const aspectDropZone = document.getElementById('aspectDropZone');
    const aspectFileInput = document.getElementById('aspectFileInput');
    const aspectDropPrompt = document.getElementById('aspectDropPrompt');
    const aspectPreviewWrapper = document.getElementById('aspectPreviewWrapper');
    const aspectThumb = document.getElementById('aspectThumb');
    const aspectFileName = document.getElementById('aspectFileName');
    const aspectFileDims = document.getElementById('aspectFileDims');
    const btnRemoveAspectFile = document.getElementById('btnRemoveAspectFile');

    const btnReuseEnhanced = document.getElementById('btnReuseEnhanced');
    const aspectSampleBtns = document.querySelectorAll('.aspect-sample-btn');
    const ratioCards = document.querySelectorAll('.ratio-card');
    const methodCards = document.querySelectorAll('.method-card');
    const padColorOptions = document.getElementById('padColorOptions');
    const colorBtns = document.querySelectorAll('.btn-color');

    const btnConvertAspect = document.getElementById('btnConvertAspect');
    const btnConvertAspectText = document.getElementById('btnConvertAspectText');
    const aspectEmptyState = document.getElementById('aspectEmptyState');
    const aspectProcessingState = document.getElementById('aspectProcessingState');
    const aspectResultContainer = document.getElementById('aspectResultContainer');

    const aspectResultImg = document.getElementById('aspectResultImg');
    const aspectRatioTag = document.getElementById('aspectRatioTag');
    const aspectRatioDimsText = document.getElementById('aspectRatioDimsText');
    const aspectStatRatio = document.getElementById('aspectStatRatio');
    const aspectStatResolution = document.getElementById('aspectStatResolution');
    const aspectStatMode = document.getElementById('aspectStatMode');

    const btnResetAspect = document.getElementById('btnResetAspect');
    const btnSaveAspectToDownloads = document.getElementById('btnSaveAspectToDownloads');
    const btnDownloadAspectPng = document.getElementById('btnDownloadAspectPng');
    const btnDownloadAspectJpg = document.getElementById('btnDownloadAspectJpg');

    let currentAspectFile = null;
    let currentAspectSourceFilename = null;
    let currentAspectSampleId = null;
    let selectedBgColor = 'black';
    let currentAspectResult = null;

    if (aspectDropZone) {
        aspectDropZone.addEventListener('click', (e) => {
            if (e.target !== btnRemoveAspectFile && !btnRemoveAspectFile.contains(e.target)) {
                aspectFileInput.click();
            }
        });

        aspectFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleAspectFile(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            aspectDropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                aspectDropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            aspectDropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                aspectDropZone.classList.remove('dragover');
            });
        });

        aspectDropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files[0]) {
                handleAspectFile(dt.files[0]);
            }
        });

        btnRemoveAspectFile.addEventListener('click', (e) => {
            e.stopPropagation();
            resetAspectInput();
        });
    }

    function handleAspectFile(file) {
        currentAspectFile = file;
        currentAspectSourceFilename = null;
        currentAspectSampleId = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            const img = new Image();
            img.onload = () => {
                aspectThumb.src = dataUrl;
                aspectFileName.textContent = file.name;
                const ratioStr = (img.naturalWidth / img.naturalHeight).toFixed(2);
                aspectFileDims.textContent = `${img.naturalWidth} x ${img.naturalHeight} (Tỷ lệ: ${ratioStr}:1)`;
                aspectDropPrompt.classList.add('hidden');
                aspectPreviewWrapper.classList.remove('hidden');
                btnConvertAspect.disabled = false;
            };
            img.src = dataUrl;
        };
        reader.readAsDataURL(file);
    }

    function resetAspectInput() {
        currentAspectFile = null;
        currentAspectSourceFilename = null;
        currentAspectSampleId = null;
        aspectFileInput.value = '';
        aspectDropPrompt.classList.remove('hidden');
        aspectPreviewWrapper.classList.add('hidden');
        btnConvertAspect.disabled = true;
    }

    // Reuse enhanced image from Tab 1
    if (btnReuseEnhanced) {
        btnReuseEnhanced.addEventListener('click', () => {
            if (currentResultData && currentResultData.enhanced_url) {
                currentAspectFile = null;
                currentAspectSampleId = null;
                currentAspectSourceFilename = currentResultData.filename;

                aspectThumb.src = currentResultData.enhanced_url;
                aspectFileName.textContent = `[Ảnh Đã Làm Nét] ${currentResultData.filename}`;
                const s = currentResultData.stats;
                aspectFileDims.textContent = `${s.final_width} x ${s.final_height} (${(s.final_width/s.final_height).toFixed(2)}:1)`;
                aspectDropPrompt.classList.add('hidden');
                aspectPreviewWrapper.classList.remove('hidden');
                btnConvertAspect.disabled = false;
                showToast('Đã nạp ảnh từ tab Làm Nét thành công!');
            } else {
                showToast('Bạn chưa có ảnh nào vừa làm nét ở Tab 1. Hãy tải ảnh mới lên nhé!', 'error');
            }
        });
    }

    // Sample buttons for aspect
    aspectSampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleType = btn.dataset.sample;
            currentAspectSampleId = sampleType;
            currentAspectFile = null;
            currentAspectSourceFilename = null;

            if (sampleType === 'anime') {
                aspectThumb.src = '/static/samples/sample_anime_lowres.jpg';
                aspectFileName.textContent = 'sample_anime_lowres.jpg (Ngang)';
                aspectFileDims.textContent = 'Ảnh mẫu Anime';
            } else {
                aspectThumb.src = '/static/samples/sample_portrait_lowres.jpg';
                aspectFileName.textContent = 'sample_portrait_lowres.jpg (Vuông)';
                aspectFileDims.textContent = 'Ảnh mẫu Chân Dung';
            }
            aspectDropPrompt.classList.add('hidden');
            aspectPreviewWrapper.classList.remove('hidden');
            btnConvertAspect.disabled = false;
        });
    });

    const cropAlignOptions = document.getElementById('cropAlignOptions');
    const alignBtns = document.querySelectorAll('.btn-align');
    let selectedAlign = 'center';

    alignBtns.forEach(b => {
        b.addEventListener('click', () => {
            alignBtns.forEach(btn => btn.classList.remove('active'));
            b.classList.add('active');
            selectedAlign = b.dataset.align;
        });
    });

    // Ratio Radio Cards
    ratioCards.forEach(card => {
        card.addEventListener('click', () => {
            ratioCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        });
    });

    // Method Radio Cards
    methodCards.forEach(card => {
        card.addEventListener('click', () => {
            methodCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const mode = card.querySelector('input').value;
            if (mode === 'pad_color') {
                padColorOptions?.classList.remove('hidden');
            } else {
                padColorOptions?.classList.add('hidden');
            }
            if (mode === 'smart_crop') {
                cropAlignOptions?.classList.remove('hidden');
            } else {
                cropAlignOptions?.classList.add('hidden');
            }
        });
    });

    // Color buttons
    colorBtns.forEach(b => {
        b.addEventListener('click', () => {
            colorBtns.forEach(btn => btn.classList.remove('active'));
            b.classList.add('active');
            selectedBgColor = b.dataset.color;
        });
    });

    // Run Aspect Conversion
    if (btnConvertAspect) {
        btnConvertAspect.addEventListener('click', async () => {
            if (!currentAspectFile && !currentAspectSourceFilename && !currentAspectSampleId) return;

            btnConvertAspect.disabled = true;
            btnConvertAspectText.textContent = 'Đang chuyển đổi...';
            aspectEmptyState.classList.add('hidden');
            aspectResultContainer.classList.add('hidden');
            aspectProcessingState.classList.remove('hidden');

            const targetRatio = document.querySelector('input[name="targetRatio"]:checked')?.value || '1:1';
            const mode = document.querySelector('input[name="aspectMode"]:checked')?.value || 'ai_expand';

            const formData = new FormData();
            formData.append('target_ratio', targetRatio);
            formData.append('mode', mode);
            formData.append('align', selectedAlign);
            formData.append('bg_color', selectedBgColor);
            formData.append('resolution_mode', 'standard');
            formData.append('bg_color', selectedBgColor);
            formData.append('resolution_mode', 'standard');

            if (currentAspectFile) {
                formData.append('image', currentAspectFile);
            } else if (currentAspectSourceFilename) {
                formData.append('source_filename', currentAspectSourceFilename);
            } else if (currentAspectSampleId) {
                formData.append('sample_id', currentAspectSampleId);
            }

            try {
                const res = await fetch('/api/convert-aspect', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    currentAspectResult = data.data;
                    displayAspectResult(data.data);
                    showToast(`Đã đổi tỷ lệ thành công trong ${data.data.stats.duration_sec}s!`);
                    fetchHistory();
                } else {
                    throw new Error(data.message || 'Lỗi xử lý ảnh.');
                }
            } catch (err) {
                console.error('Aspect conversion error:', err);
                showToast(`Lỗi: ${err.message}`, 'error');
                aspectProcessingState.classList.add('hidden');
                aspectEmptyState.classList.remove('hidden');
            } finally {
                btnConvertAspect.disabled = false;
                btnConvertAspectText.textContent = 'Chuyển Đổi Tỷ Lệ Ngay';
            }
        });
    }

    function displayAspectResult(data) {
        aspectProcessingState.classList.add('hidden');
        aspectResultContainer.classList.remove('hidden');

        aspectResultImg.src = data.result_url;
        const s = data.stats;
        aspectRatioTag.textContent = s.ratio_name;
        aspectRatioDimsText.textContent = `${s.orig_width}x${s.orig_height} ➔ ${s.final_width}x${s.final_height}`;

        aspectStatRatio.textContent = s.target_ratio;
        aspectStatResolution.textContent = `${s.final_width} x ${s.final_height}`;
        const modeLabels = {
            'ai_expand': '✨ AI Sinh Không Gian Mới (Outpaint AI - Giữ trọn 100% ảnh)',
            'smart_crop': 'Cắt tràn khung hình (Zoom & Crop)',
            'stretch': 'Co giãn tràn khung (Stretch)',
            'blur_expand': 'Mở rộng nền mờ (Ambient Blur)',
            'pad_color': 'Khung viền màu (Fit & Pad)'
        };
        aspectStatMode.textContent = modeLabels[s.mode] || s.mode;
        aspectResultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    if (btnResetAspect) {
        btnResetAspect.addEventListener('click', () => {
            aspectResultContainer.classList.add('hidden');
            aspectEmptyState.classList.remove('hidden');
            resetAspectInput();
        });
    }

    // Aspect Download Actions
    if (btnSaveAspectToDownloads) {
        btnSaveAspectToDownloads.addEventListener('click', async () => {
            if (!currentAspectResult) return;
            const filename = currentAspectResult.filename;
            showToast('Đang lưu tệp vào thư mục Downloads...');
            try {
                const formData = new FormData();
                formData.append('filename', filename);
                const res = await fetch('/api/save-to-downloads', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp vào Downloads: ${data.filename}`);
                } else {
                    throw new Error(data.message);
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    if (btnDownloadAspectPng) {
        btnDownloadAspectPng.addEventListener('click', () => {
            if (!currentAspectResult) return;
            const filename = `ClarityAI_${currentAspectResult.filename}`;
            triggerDownload(currentAspectResult.download_url, filename);
        });
    }

    if (btnDownloadAspectJpg) {
        btnDownloadAspectJpg.addEventListener('click', () => {
            if (!currentAspectResult) return;
            const filename = `ClarityAI_${currentAspectResult.filename.replace('.png', '.jpg')}`;
            triggerDownload(currentAspectResult.download_jpg_url, filename);
        });
    }

    // ==========================================
    // 12. BATCH RESIZE STUDIO CLIENT LOGIC (NEW FEATURE)
    // ==========================================
    const batchDropZone = document.getElementById('batchDropZone');
    const batchFileInput = document.getElementById('batchFileInput');
    const batchDropPrompt = document.getElementById('batchDropPrompt');
    const batchFilesListWrapper = document.getElementById('batchFilesListWrapper');
    const batchSelectedCount = document.getElementById('batchSelectedCount');
    const btnClearBatchFiles = document.getElementById('btnClearBatchFiles');
    const batchFilesGrid = document.getElementById('batchFilesGrid');
    const btnAddMoreFiles = document.getElementById('btnAddMoreFiles');
    const batchFileCountBadge = document.getElementById('batchFileCountBadge');

    const batchTargetWidth = document.getElementById('batchTargetWidth');
    const batchTargetHeight = document.getElementById('batchTargetHeight');
    const btnSwapDims = document.getElementById('btnSwapDims');
    const presetTags = document.querySelectorAll('.preset-tag');
    const resizeModeCards = document.querySelectorAll('.resize-mode-card');

    const btnRunBatchResize = document.getElementById('btnRunBatchResize');
    const btnRunBatchResizeText = document.getElementById('btnRunBatchResizeText');

    const batchEmptyState = document.getElementById('batchEmptyState');
    const batchProcessingState = document.getElementById('batchProcessingState');
    const batchResultContainer = document.getElementById('batchResultContainer');
    const batchProcessTitle = document.getElementById('batchProcessTitle');
    const batchProcessSub = document.getElementById('batchProcessSub');
    const batchProgressBar = document.getElementById('batchProgressBar');
    const batchProgressPercent = document.getElementById('batchProgressPercent');

    const batchZipTitle = document.getElementById('batchZipTitle');
    const batchResultCount = document.getElementById('batchResultCount');
    const batchResultDims = document.getElementById('batchResultDims');
    const batchResultZipSize = document.getElementById('batchResultZipSize');
    const batchResultCountText = document.getElementById('batchResultCountText');
    const batchResultsGrid = document.getElementById('batchResultsGrid');

    const btnDownloadBatchZip = document.getElementById('btnDownloadBatchZip');
    const btnSaveBatchZipToDownloads = document.getElementById('btnSaveBatchZipToDownloads');
    const btnResetBatch = document.getElementById('btnResetBatch');

    // State Variables for Batch
    let selectedBatchFiles = [];
    let isBatchProcessing = false;
    let currentBatchResult = null;
    let batchPreviewUrls = [];

    // 1. File Upload & Trigger
    if (batchDropZone) {
        batchDropZone.addEventListener('click', (e) => {
            if (e.target !== btnClearBatchFiles && !btnClearBatchFiles.contains(e.target)) {
                batchFileInput.click();
            }
        });

        if (btnAddMoreFiles) {
            btnAddMoreFiles.addEventListener('click', (e) => {
                e.stopPropagation();
                batchFileInput.click();
            });
        }

        batchFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                appendBatchFiles(Array.from(e.target.files));
                batchFileInput.value = ''; // Reset input so same files can be re-selected if needed
            }
        });

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(evt => {
            batchDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                batchDropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            batchDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                batchDropZone.classList.remove('dragover');
            });
        });

        batchDropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                appendBatchFiles(Array.from(dt.files));
            }
        });
    }

    // Clipboard Paste for Batch when in batch view
    document.addEventListener('paste', (e) => {
        if (!viewBatchResize || viewBatchResize.classList.contains('hidden')) return;
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;

        const pastedFiles = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                if (blob) {
                    const ext = blob.type.split('/')[1] || 'png';
                    const fileName = `pasted_batch_${Date.now()}_${i + 1}.${ext}`;
                    const file = new File([blob], fileName, { type: blob.type });
                    pastedFiles.push(file);
                }
            }
        }
        if (pastedFiles.length > 0) {
            appendBatchFiles(pastedFiles);
            showToast(`📋 Đã dán ${pastedFiles.length} ảnh từ Clipboard!`);
        }
    });

    // Append files to batch list (avoiding exact duplicates)
    function appendBatchFiles(newFiles) {
        const validExtensions = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
        let addedCount = 0;

        newFiles.forEach(file => {
            if (!validExtensions.includes(file.type) && !file.name.match(/\.(png|jpe?g|webp|bmp|jfif)$/i)) {
                return;
            }
            // Check duplicate by name and size
            const isDuplicate = selectedBatchFiles.some(f => f.name === file.name && f.size === file.size);
            if (!isDuplicate) {
                selectedBatchFiles.push(file);
                addedCount++;
            }
        });

        if (addedCount > 0) {
            renderBatchFilesList();
            showToast(`Đã thêm ${addedCount} ảnh vào danh sách resize.`);
        }
    }

    // Render file preview cards in Step 1
    function renderBatchFilesList() {
        // Clean up previous object URLs
        batchPreviewUrls.forEach(url => URL.revokeObjectURL(url));
        batchPreviewUrls = [];

        const count = selectedBatchFiles.length;

        if (count === 0) {
            batchFilesListWrapper.classList.add('hidden');
            batchDropPrompt.classList.remove('hidden');
            batchFileCountBadge.classList.add('hidden');
            btnRunBatchResize.disabled = true;
            btnRunBatchResizeText.textContent = 'Bắt Đầu Resize Hàng Loạt (0 ảnh)';
            return;
        }

        batchFilesListWrapper.classList.remove('hidden');
        batchDropPrompt.classList.add('hidden');
        batchFileCountBadge.classList.remove('hidden');
        batchFileCountBadge.textContent = `${count} ảnh`;
        batchSelectedCount.textContent = count;

        btnRunBatchResize.disabled = false;
        btnRunBatchResizeText.textContent = `Bắt Đầu Resize Hàng Loạt (${count} ảnh)`;

        // Render mini grid
        batchFilesGrid.innerHTML = '';
        selectedBatchFiles.forEach((file, index) => {
            const blobUrl = URL.createObjectURL(file);
            batchPreviewUrls.push(blobUrl);

            const card = document.createElement('div');
            card.className = 'batch-file-card';

            const sizeText = file.size > 1024 * 1024 
                ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                : `${Math.round(file.size / 1024)} KB`;

            card.innerHTML = `
                <img src="${blobUrl}" alt="${file.name}" class="batch-file-thumb">
                <div class="batch-file-name" title="${file.name}">${file.name}</div>
                <div class="batch-file-meta">
                    <span>#${index + 1}</span>
                    <span>${sizeText}</span>
                </div>
                <button type="button" class="btn-remove-batch-item" title="Xóa ảnh này" data-index="${index}">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            `;

            // Remove single item listener
            const removeBtn = card.querySelector('.btn-remove-batch-item');
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                selectedBatchFiles.splice(index, 1);
                renderBatchFilesList();
            });

            batchFilesGrid.appendChild(card);
        });
    }

    // Clear all batch files
    if (btnClearBatchFiles) {
        btnClearBatchFiles.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedBatchFiles = [];
            renderBatchFilesList();
            showToast('Đã xóa tất cả ảnh đã chọn.');
        });
    }

    // Swap Width and Height
    if (btnSwapDims) {
        btnSwapDims.addEventListener('click', () => {
            const currentW = batchTargetWidth.value;
            const currentH = batchTargetHeight.value;
            batchTargetWidth.value = currentH;
            batchTargetHeight.value = currentW;
            syncPresetState();
            showToast(`Đã hoán đổi kích thước: ${batchTargetWidth.value} x ${batchTargetHeight.value} px`);
        });
    }

    // Preset Dimensions Buttons
    presetTags.forEach(btn => {
        btn.addEventListener('click', () => {
            presetTags.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const w = btn.getAttribute('data-w');
            const h = btn.getAttribute('data-h');
            batchTargetWidth.value = w;
            batchTargetHeight.value = h;
        });
    });

    [batchTargetWidth, batchTargetHeight].forEach(inp => {
        if (inp) {
            inp.addEventListener('input', () => {
                syncPresetState();
            });
        }
    });

    function syncPresetState() {
        const w = batchTargetWidth.value;
        const h = batchTargetHeight.value;
        presetTags.forEach(btn => {
            if (btn.getAttribute('data-w') === w && btn.getAttribute('data-h') === h) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    // Resize Mode Radio Cards
    resizeModeCards.forEach(card => {
        card.addEventListener('click', () => {
            resizeModeCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    // 2. Run Batch Resize
    if (btnRunBatchResize) {
        btnRunBatchResize.addEventListener('click', async () => {
            if (selectedBatchFiles.length === 0 || isBatchProcessing) return;

            const targetW = parseInt(batchTargetWidth.value, 10);
            const targetH = parseInt(batchTargetHeight.value, 10);

            if (isNaN(targetW) || isNaN(targetH) || targetW < 16 || targetH < 16 || targetW > 12000 || targetH > 12000) {
                showToast('Vui lòng nhập chiều rộng và chiều cao hợp lệ (từ 16 đến 12000 px).', 'error');
                return;
            }

            const selectedMode = document.querySelector('input[name="batchResizeMode"]:checked')?.value || 'stretch';

            // Switch to Processing State
            isBatchProcessing = true;
            btnRunBatchResize.disabled = true;
            batchEmptyState.classList.add('hidden');
            batchResultContainer.classList.add('hidden');
            batchProcessingState.classList.remove('hidden');

            batchProcessTitle.textContent = `Đang Resize ${selectedBatchFiles.length} Ảnh...`;
            batchProcessSub.textContent = `Hệ thống đang chuyển đổi tất cả ảnh sang chuẩn PNG ${targetW}x${targetH} (không viền đen) và nén ZIP...`;
            batchProgressBar.style.width = '15%';
            batchProgressPercent.textContent = 'Đang tải tệp lên server...';

            // Progress bar simulation
            let prog = 15;
            const progressInterval = setInterval(() => {
                if (prog < 88) {
                    prog += Math.floor(Math.random() * 8) + 4;
                    if (prog > 88) prog = 88;
                    batchProgressBar.style.width = `${prog}%`;
                    batchProgressPercent.textContent = `Đang xử lý ${prog}%...`;
                }
            }, 300);

            try {
                const formData = new FormData();
                formData.append('width', targetW);
                formData.append('height', targetH);
                formData.append('mode', selectedMode);

                selectedBatchFiles.forEach(file => {
                    formData.append('images', file);
                });

                const res = await fetch('/api/batch-resize', {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                const data = await res.json();

                if (data.status === 'success') {
                    batchProgressBar.style.width = '100%';
                    batchProgressPercent.textContent = 'Hoàn tất 100%!';

                    setTimeout(() => {
                        currentBatchResult = data;
                        displayBatchResults(data);
                        showToast(`✅ Đã resize thành công ${data.total_files} ảnh sang PNG! Nhấn Tải Về Trọn Bộ để nhận file ZIP.`);
                    }, 400);
                } else {
                    throw new Error(data.message || 'Xử lý thất bại');
                }
            } catch (err) {
                clearInterval(progressInterval);
                showToast(`Lỗi: ${err.message}`, 'error');
                batchProcessingState.classList.add('hidden');
                batchEmptyState.classList.remove('hidden');
            } finally {
                isBatchProcessing = false;
                btnRunBatchResize.disabled = false;
            }
        });
    }

    // Display Batch Results
    function displayBatchResults(data) {
        batchProcessingState.classList.add('hidden');
        batchResultContainer.classList.remove('hidden');

        // ZIP Card Info
        batchZipTitle.textContent = data.zip_filename;
        batchResultCount.textContent = data.total_files;
        batchResultDims.textContent = `${data.target_w} x ${data.target_h} px`;
        batchResultZipSize.textContent = `${data.zip_size_mb} MB`;
        batchResultCountText.textContent = data.total_files;

        // Render images grid
        batchResultsGrid.innerHTML = '';
        data.images.forEach((item, idx) => {
            const card = document.createElement('div');
            card.className = 'batch-result-card';
            card.innerHTML = `
                <div class="batch-result-img-box">
                    <img src="${item.url}?t=${Date.now()}" alt="${item.filename}" class="batch-result-img" loading="lazy">
                </div>
                <div class="batch-result-info">
                    <div class="batch-result-filename" title="${item.filename}">${item.filename}</div>
                    <div class="batch-result-dims">
                        <i class="fa-solid fa-check text-cyan"></i>
                        <span>${item.orig_w}x${item.orig_h} ➔ <strong>${item.target_w}x${item.target_h}</strong></span>
                    </div>
                    <div class="batch-result-bottom">
                        <span class="batch-result-filesize">${item.size_kb} KB · PNG</span>
                        <div class="batch-card-actions">
                            <button type="button" class="btn-download-single-png" title="Tải riêng tệp PNG qua trình duyệt">
                                <i class="fa-solid fa-download"></i> Tải PNG
                            </button>
                            <button type="button" class="btn-save-single-downloads" title="Lưu trực tiếp vào thư mục Downloads của Windows">
                                <i class="fa-solid fa-folder-arrow-down"></i> Lưu Downloads
                            </button>
                        </div>
                    </div>
                </div>
            `;

            // Single image download button
            const singleBtn = card.querySelector('.btn-download-single-png');
            singleBtn.addEventListener('click', () => {
                triggerDownload(item.download_url, item.filename);
            });

            // Single image save to downloads button
            const saveBtn = card.querySelector('.btn-save-single-downloads');
            saveBtn.addEventListener('click', async () => {
                showToast(`Đang lưu ảnh ${item.filename} vào Downloads...`);
                try {
                    const fd = new FormData();
                    fd.append('batch_id', item.batch_id || data.batch_id);
                    fd.append('filename', item.filename);
                    const res = await fetch('/api/save-batch-item-to-downloads', {
                        method: 'POST',
                        body: fd
                    });
                    const d = await res.json();
                    if (d.status === 'success') {
                        showToast(`✅ Đã lưu trực tiếp vào Downloads: ${d.filename}`);
                    } else {
                        throw new Error(d.message);
                    }
                } catch (e) {
                    showToast(`Lỗi: ${e.message}`, 'error');
                }
            });

            batchResultsGrid.appendChild(card);
        });

        // Scroll smoothly to results
        batchResultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Download ZIP Action
    if (btnDownloadBatchZip) {
        btnDownloadBatchZip.addEventListener('click', () => {
            if (!currentBatchResult) return;
            triggerDownload(currentBatchResult.zip_url, currentBatchResult.zip_filename);
            showToast(`📦 Đang tải file ZIP: ${currentBatchResult.zip_filename}`);
        });
    }

    // Save ZIP Directly to Downloads Folder
    if (btnSaveBatchZipToDownloads) {
        btnSaveBatchZipToDownloads.addEventListener('click', async () => {
            if (!currentBatchResult) return;
            const filename = currentBatchResult.zip_filename;
            showToast('Đang lưu tệp ZIP vào thư mục Downloads của máy...');

            try {
                const formData = new FormData();
                formData.append('filename', filename);
                const res = await fetch('/api/save-batch-zip-to-downloads', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp file ZIP vào Downloads: ${data.filename}`);
                } else {
                    throw new Error(data.message);
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    // Reset Batch Studio
    if (btnResetBatch) {
        btnResetBatch.addEventListener('click', () => {
            batchResultContainer.classList.add('hidden');
            batchEmptyState.classList.remove('hidden');
            selectedBatchFiles = [];
            renderBatchFilesList();
        });
    }

    // ==========================================
    // 13. SMART TARGET-MB COMPRESSOR STUDIO LOGIC
    // ==========================================
    const compressDropZone = document.getElementById('compressDropZone');
    const compressFileInput = document.getElementById('compressFileInput');
    const compressDropPrompt = document.getElementById('compressDropPrompt');
    const compressPreviewWrapper = document.getElementById('compressPreviewWrapper');
    const compressThumb = document.getElementById('compressThumb');
    const compressFileName = document.getElementById('compressFileName');
    const compressFileDims = document.getElementById('compressFileDims');
    const btnRemoveCompressFile = document.getElementById('btnRemoveCompressFile');

    // Multi-file Preview Elements
    const compressFilesListWrapper = document.getElementById('compressFilesListWrapper');
    const compressSelectedCount = document.getElementById('compressSelectedCount');
    const btnClearCompressFiles = document.getElementById('btnClearCompressFiles');
    const compressFilesGrid = document.getElementById('compressFilesGrid');
    const btnAddMoreCompressFiles = document.getElementById('btnAddMoreCompressFiles');

    const btnReuseForCompress = document.getElementById('btnReuseForCompress');
    const compressSampleBtns = document.querySelectorAll('.compress-sample-btn');

    const targetMbInput = document.getElementById('targetMbInput');
    const presetTagsMb = document.querySelectorAll('.preset-tag-mb');
    const compressFmtCards = document.querySelectorAll('.compress-fmt-card');

    const btnRunCompress = document.getElementById('btnRunCompress');
    const btnRunCompressText = document.getElementById('btnRunCompressText');

    const compressEmptyState = document.getElementById('compressEmptyState');
    const compressProcessingState = document.getElementById('compressProcessingState');
    const compressResultContainer = document.getElementById('compressResultContainer');
    const compressProcessTitle = document.getElementById('compressProcessTitle');
    const compressProcessSub = document.getElementById('compressProcessSub');

    // Batch Result Elements for Compressor
    const compressBatchResultContainer = document.getElementById('compressBatchResultContainer');
    const compressZipTitle = document.getElementById('compressZipTitle');
    const compressBatchResultCount = document.getElementById('compressBatchResultCount');
    const compressBatchReduction = document.getElementById('compressBatchReduction');
    const compressBatchZipSize = document.getElementById('compressBatchZipSize');
    const compressBatchCountText = document.getElementById('compressBatchCountText');
    const btnDownloadCompressZip = document.getElementById('btnDownloadCompressZip');
    const btnSaveCompressZipToDownloads = document.getElementById('btnSaveCompressZipToDownloads');
    const btnResetCompressBatch = document.getElementById('btnResetCompressBatch');
    const compressResultsGrid = document.getElementById('compressResultsGrid');

    // Result & Slider Elements (for Single View)
    const compressComparisonStage = document.getElementById('compressComparisonStage');
    const compressSliderWrapper = document.getElementById('compressSliderWrapper');
    const compressSliderClip = document.getElementById('compressSliderClip');
    const compressSliderHandle = document.getElementById('compressSliderHandle');
    const compressImgResult = document.getElementById('compressImgResult');
    const compressImgOriginal = document.getElementById('compressImgOriginal');
    const compressTagBefore = document.getElementById('compressTagBefore');
    const compressTagAfter = document.getElementById('compressTagAfter');

    // Stats Bar Elements
    const compressStatOrigSize = document.getElementById('compressStatOrigSize');
    const compressStatNewSize = document.getElementById('compressStatNewSize');
    const compressStatReduction = document.getElementById('compressStatReduction');
    const compressStatQuality = document.getElementById('compressStatQuality');
    const compressStatDuration = document.getElementById('compressStatDuration');

    // View Tabs & Zoom
    const compressViewTabs = document.querySelectorAll('[data-compress-mode]');
    const btnCompressZoomFit = document.getElementById('btnCompressZoomFit');
    const btnCompressZoomToggle = document.getElementById('btnCompressZoomToggle');

    // Single Actions
    const btnDownloadCompressed = document.getElementById('btnDownloadCompressed');
    const btnSaveCompressToDownloads = document.getElementById('btnSaveCompressToDownloads');
    const btnResetCompress = document.getElementById('btnResetCompress');

    // State Variables for Compressor
    let selectedCompressFiles = [];
    let currentCompressSourceFilename = null;
    let currentCompressSampleId = null;
    let isCompressProcessing = false;
    let currentCompressBatchData = null;
    let currentSingleCompressItem = null;
    let isCompressDragging = false;
    let isCompressZoomed = false;

    // File Upload & Drag-and-Drop
    if (compressDropZone) {
        compressDropZone.addEventListener('click', (e) => {
            if (e.target !== btnRemoveCompressFile && !btnRemoveCompressFile.contains(e.target)) {
                compressFileInput.click();
            }
        });

        compressFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleCompressFilesSelect(e.target.files);
            }
        });

        ['dragenter', 'dragover'].forEach(evt => {
            compressDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                compressDropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            compressDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                compressDropZone.classList.remove('dragover');
            });
        });

        compressDropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleCompressFilesSelect(dt.files);
            }
        });
    }

    if (btnAddMoreCompressFiles) {
        btnAddMoreCompressFiles.addEventListener('click', (e) => {
            e.stopPropagation();
            compressFileInput.click();
        });
    }

    if (btnClearCompressFiles) {
        btnClearCompressFiles.addEventListener('click', (e) => {
            e.stopPropagation();
            resetCompressInput();
        });
    }

    // Clipboard Paste for Compressor
    document.addEventListener('paste', (e) => {
        if (!viewCompressor || viewCompressor.classList.contains('hidden')) return;
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;

        const pastedFiles = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                if (blob) {
                    const ext = blob.type.split('/')[1] || 'png';
                    const file = new File([blob], `pasted_compress_${Date.now()}_${i}.${ext}`, { type: blob.type });
                    pastedFiles.push(file);
                }
            }
        }
        if (pastedFiles.length > 0) {
            handleCompressFilesSelect(pastedFiles);
            showToast(`📋 Đã dán ${pastedFiles.length} ảnh từ Clipboard để nén!`);
        }
    });

    function handleCompressFilesSelect(filesList) {
        currentCompressSourceFilename = null;
        currentCompressSampleId = null;

        const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        const newFiles = Array.from(filesList).filter(f => {
            return allowed.some(type => f.type.includes(type.split('/')[1])) || f.name.match(/\.(jpg|jpeg|png|webp|bmp|jfif)$/i);
        });

        if (newFiles.length === 0) {
            showToast('Vui lòng chọn các tệp ảnh hợp lệ (PNG, JPG, WebP, BMP).', 'error');
            return;
        }

        selectedCompressFiles = selectedCompressFiles.concat(newFiles);

        if (selectedCompressFiles.length === 1) {
            // Hiển thị dạng 1 ảnh
            const file = selectedCompressFiles[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                compressThumb.src = e.target.result;
                compressFileName.textContent = file.name;
                const sizeStr = file.size > 1024 * 1024 
                    ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` 
                    : `${Math.round(file.size / 1024)} KB`;

                const img = new Image();
                img.onload = () => {
                    compressFileDims.textContent = `${img.width} x ${img.height} · ${sizeStr}`;
                };
                img.src = e.target.result;

                compressDropPrompt.classList.add('hidden');
                compressPreviewWrapper.classList.remove('hidden');
                compressFilesListWrapper.classList.add('hidden');
                updateCompressButtonState();
            };
            reader.readAsDataURL(file);
        } else {
            // Hiển thị danh sách nhiều ảnh
            compressDropPrompt.classList.add('hidden');
            compressPreviewWrapper.classList.add('hidden');
            compressFilesListWrapper.classList.remove('hidden');
            renderCompressFilesList();
            updateCompressButtonState();
        }
    }

    function renderCompressFilesList() {
        if (!compressFilesGrid) return;
        compressFilesGrid.innerHTML = '';
        compressSelectedCount.textContent = selectedCompressFiles.length;

        selectedCompressFiles.forEach((file, index) => {
            const card = document.createElement('div');
            card.className = 'batch-file-mini-card';
            const sizeStr = file.size > 1024 * 1024 
                ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` 
                : `${Math.round(file.size / 1024)} KB`;

            card.innerHTML = `
                <img src="" alt="${file.name}" class="mini-thumb">
                <div class="mini-meta">
                    <span class="mini-name" title="${file.name}">${file.name}</span>
                    <span class="mini-size">${sizeStr}</span>
                </div>
                <button type="button" class="btn-remove-mini" title="Bỏ ảnh này">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            `;

            const imgEl = card.querySelector('.mini-thumb');
            const reader = new FileReader();
            reader.onload = (e) => { imgEl.src = e.target.result; };
            reader.readAsDataURL(file);

            const removeBtn = card.querySelector('.btn-remove-mini');
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                selectedCompressFiles.splice(index, 1);
                if (selectedCompressFiles.length === 0) {
                    resetCompressInput();
                } else if (selectedCompressFiles.length === 1) {
                    handleCompressFilesSelect(selectedCompressFiles);
                } else {
                    renderCompressFilesList();
                    updateCompressButtonState();
                }
            });

            compressFilesGrid.appendChild(card);
        });
    }

    if (btnRemoveCompressFile) {
        btnRemoveCompressFile.addEventListener('click', (e) => {
            e.stopPropagation();
            resetCompressInput();
        });
    }

    function resetCompressInput() {
        selectedCompressFiles = [];
        currentCompressSourceFilename = null;
        currentCompressSampleId = null;
        compressFileInput.value = '';
        compressThumb.src = '';
        compressDropPrompt.classList.remove('hidden');
        compressPreviewWrapper.classList.add('hidden');
        compressFilesListWrapper.classList.add('hidden');
        updateCompressButtonState();
    }

    // Reuse from previous sessions/tabs
    if (btnReuseForCompress) {
        btnReuseForCompress.addEventListener('click', () => {
            if (currentResultData && currentResultData.filename) {
                selectedCompressFiles = [];
                currentCompressSourceFilename = currentResultData.filename;
                currentCompressSampleId = null;

                compressThumb.src = currentResultData.enhanced_url;
                compressFileName.textContent = `Vừa làm nét: ${currentResultData.filename}`;
                compressFileDims.textContent = `${currentResultData.stats.final_width} x ${currentResultData.stats.final_height} · ${currentResultData.file_sizes.enhanced_mb} MB`;

                compressDropPrompt.classList.add('hidden');
                compressPreviewWrapper.classList.remove('hidden');
                compressFilesListWrapper.classList.add('hidden');
                updateCompressButtonState();
                showToast('Đã nạp ảnh vừa làm nét để nén.');
            } else {
                showToast('Chưa có ảnh nào vừa được xử lý gần đây!', 'error');
            }
        });
    }

    // Samples for Compressor
    compressSampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleId = btn.getAttribute('data-sample');
            currentCompressSampleId = sampleId;
            selectedCompressFiles = [];
            currentCompressSourceFilename = null;

            const sampleUrl = sampleId === 'portrait' ? '/static/samples/sample_portrait_lowres.jpg' : '/static/samples/sample_anime_lowres.jpg';
            compressThumb.src = sampleUrl;
            compressFileName.textContent = `Ảnh mẫu: ${sampleId}.jpg`;
            compressFileDims.textContent = 'Ảnh mẫu thử nghiệm';

            compressDropPrompt.classList.add('hidden');
            compressPreviewWrapper.classList.remove('hidden');
            compressFilesListWrapper.classList.add('hidden');
            updateCompressButtonState();
            showToast(`Đã chọn ảnh mẫu ${sampleId}.`);
        });
    });

    // Preset MB buttons
    presetTagsMb.forEach(btn => {
        btn.addEventListener('click', () => {
            presetTagsMb.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mb = btn.getAttribute('data-mb');
            targetMbInput.value = mb;
            updateCompressButtonState();
        });
    });

    if (targetMbInput) {
        targetMbInput.addEventListener('input', () => {
            const val = parseFloat(targetMbInput.value);
            presetTagsMb.forEach(b => {
                if (parseFloat(b.getAttribute('data-mb')) === val) {
                    b.classList.add('active');
                } else {
                    b.classList.remove('active');
                }
            });
            updateCompressButtonState();
        });
    }

    // Format selection
    compressFmtCards.forEach(card => {
        card.addEventListener('click', () => {
            compressFmtCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    function updateCompressButtonState() {
        const count = selectedCompressFiles.length || (currentCompressSourceFilename || currentCompressSampleId ? 1 : 0);
        const targetMb = parseFloat(targetMbInput.value) || 1.0;

        if (count > 1) {
            btnRunCompress.disabled = false;
            btnRunCompressText.textContent = `Bắt Đầu Nén Hàng Loạt (${count} ảnh) Xuống ${targetMb} MB`;
        } else if (count === 1) {
            btnRunCompress.disabled = false;
            btnRunCompressText.textContent = `Bắt Đầu Nén Xuống ${targetMb} MB`;
        } else {
            btnRunCompress.disabled = true;
            btnRunCompressText.textContent = `Bắt Đầu Nén Xuống ${targetMb} MB`;
        }
    }

    // Run Compression (Supports Single & Batch)
    if (btnRunCompress) {
        btnRunCompress.addEventListener('click', async () => {
            if (isCompressProcessing) return;
            const targetMb = parseFloat(targetMbInput.value);
            if (isNaN(targetMb) || targetMb < 0.02 || targetMb > 500) {
                showToast('Vui lòng nhập mức MB hợp lệ (từ 0.05 đến 100 MB).', 'error');
                return;
            }

            const selectedFmt = document.querySelector('input[name="compressFormat"]:checked')?.value || 'auto';
            const count = selectedCompressFiles.length || (currentCompressSourceFilename || currentCompressSampleId ? 1 : 0);

            if (count === 0) {
                showToast('Vui lòng chọn ít nhất 1 ảnh để nén.', 'error');
                return;
            }

            // Show Processing State
            isCompressProcessing = true;
            btnRunCompress.disabled = true;
            compressEmptyState.classList.add('hidden');
            compressResultContainer.classList.add('hidden');
            compressBatchResultContainer.classList.add('hidden');
            compressProcessingState.classList.remove('hidden');

            compressProcessTitle.textContent = count > 1 
                ? `Đang Nén Hàng Loạt ${count} Ảnh Xuống ${targetMb} MB...`
                : `Đang Nén Xuống Mức ${targetMb} MB...`;
            compressProcessSub.textContent = `Giữ nguyên 100% kích thước gốc (Width x Height), bảo toàn chi tiết và tên ảnh gốc...`;

            try {
                const formData = new FormData();
                formData.append('target_mb', targetMb);
                formData.append('format_choice', selectedFmt);

                if (selectedCompressFiles.length > 0) {
                    selectedCompressFiles.forEach(file => {
                        formData.append('images', file);
                    });
                } else if (currentCompressSourceFilename) {
                    formData.append('source_filename', currentCompressSourceFilename);
                } else if (currentCompressSampleId) {
                    formData.append('sample_id', currentCompressSampleId);
                }

                const res = await fetch('/api/batch-compress', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.status === 'success') {
                    currentCompressBatchData = data;

                    if (data.total_files > 1) {
                        displayBatchCompressResult(data);
                        showToast(`✅ Đã nén thành công trọn bộ ${data.total_files} ảnh! Giảm ${data.total_reduction_percent}%.`);
                    } else if (data.total_files === 1 && data.images[0]) {
                        // 1 ảnh: hiển thị thanh trượt so sánh Before/After chi tiết
                        displaySingleCompressResult(data.images[0], data);
                        showToast(`✅ Đã nén thành công! Giảm ${data.images[0].reduction_percent}% (Đạt ${data.images[0].final_size_mb} MB).`);
                    }
                } else {
                    throw new Error(data.message || 'Nén ảnh thất bại');
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
                compressProcessingState.classList.add('hidden');
                compressEmptyState.classList.remove('hidden');
            } finally {
                isCompressProcessing = false;
                btnRunCompress.disabled = false;
            }
        });
    }

    // Display Batch Compress Result Gallery
    function displayBatchCompressResult(data) {
        compressProcessingState.classList.add('hidden');
        compressResultContainer.classList.add('hidden');
        compressBatchResultContainer.classList.remove('hidden');

        compressZipTitle.textContent = data.zip_filename;
        compressBatchResultCount.textContent = data.total_files;
        compressBatchReduction.textContent = `${data.total_reduction_percent}%`;
        compressBatchZipSize.textContent = `${data.zip_size_mb} MB`;
        compressBatchCountText.textContent = data.total_files;

        // Render Cards in Grid
        compressResultsGrid.innerHTML = '';
        data.images.forEach(item => {
            const card = document.createElement('div');
            card.className = 'batch-result-card';
            card.innerHTML = `
                <div class="batch-result-img-box">
                    <img src="${item.url}?t=${Date.now()}" alt="${item.filename}" class="batch-result-img" loading="lazy">
                </div>
                <div class="batch-result-info">
                    <div class="batch-result-filename" title="${item.filename}">${item.filename}</div>
                    <div class="batch-result-dims">
                        <i class="fa-solid fa-lock text-cyan"></i>
                        <span>${item.orig_w}x${item.orig_h} (100% Nguyên Bản)</span>
                    </div>
                    <div class="batch-result-bottom">
                        <span class="batch-result-filesize">${item.orig_size_mb} MB ➔ <strong>${item.final_size_mb} MB</strong> (-${item.reduction_percent}%)</span>
                        <div class="batch-card-actions">
                            <button type="button" class="btn-download-single-file" title="Tải lẻ ảnh này về máy">
                                <i class="fa-solid fa-download"></i> Tải Lẻ
                            </button>
                            <button type="button" class="btn-save-single-downloads" title="Lưu trực tiếp vào thư mục Downloads của Windows">
                                <i class="fa-solid fa-folder-arrow-down"></i> Lưu Downloads
                            </button>
                        </div>
                    </div>
                </div>
            `;

            // Download single file
            const dlBtn = card.querySelector('.btn-download-single-file');
            dlBtn.addEventListener('click', () => {
                triggerDownload(item.download_url, item.filename);
            });

            // Save single file directly to Windows Downloads
            const saveBtn = card.querySelector('.btn-save-single-downloads');
            saveBtn.addEventListener('click', async () => {
                showToast(`Đang lưu ảnh ${item.filename} vào Downloads...`);
                try {
                    const fd = new FormData();
                    fd.append('batch_id', item.batch_id || data.batch_id);
                    fd.append('filename', item.filename);
                    const res = await fetch('/api/save-compress-item-to-downloads', {
                        method: 'POST',
                        body: fd
                    });
                    const resData = await res.json();
                    if (resData.status === 'success') {
                        showToast(`✅ Đã lưu trực tiếp vào Downloads: ${resData.filename}`);
                    } else {
                        throw new Error(resData.message);
                    }
                } catch (e) {
                    showToast(`Lỗi: ${e.message}`, 'error');
                }
            });

            compressResultsGrid.appendChild(card);
        });

        compressBatchResultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Display Single Compress Result with Interactive Slider
    function displaySingleCompressResult(item, batchData) {
        currentSingleCompressItem = item;
        compressProcessingState.classList.add('hidden');
        compressBatchResultContainer.classList.add('hidden');
        compressResultContainer.classList.remove('hidden');

        // Image URLs
        compressImgResult.src = `${item.url}?t=${Date.now()}`;
        compressImgOriginal.src = `${item.url}?t=${Date.now()}`; // Preview compressed image in slider

        // Tag text
        compressTagBefore.textContent = `GỐC (${item.orig_size_mb} MB)`;
        compressTagAfter.textContent = `NÉN (${item.final_size_mb} MB)`;

        // Stats Bar
        compressStatOrigSize.textContent = `${item.orig_size_mb} MB`;
        compressStatNewSize.textContent = `${item.final_size_mb} MB`;
        compressStatReduction.textContent = `Giảm -${item.reduction_percent}%`;
        compressStatQuality.textContent = `${item.orig_w}x${item.orig_h} (100% Nguyên Bản) · Q=${item.quality_used}`;
        compressStatDuration.textContent = `⚡ 100% Kích Thước Gốc`;

        setCompressSliderPos(50);
        compressResultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ZIP Download for Batch Compress
    if (btnDownloadCompressZip) {
        btnDownloadCompressZip.addEventListener('click', () => {
            if (!currentCompressBatchData) return;
            triggerDownload(currentCompressBatchData.zip_url, currentCompressBatchData.zip_filename);
            showToast(`📦 Đang tải file ZIP: ${currentCompressBatchData.zip_filename}`);
        });
    }

    // Save ZIP Directly to Downloads Folder
    if (btnSaveCompressZipToDownloads) {
        btnSaveCompressZipToDownloads.addEventListener('click', async () => {
            if (!currentCompressBatchData) return;
            showToast('Đang lưu tệp ZIP vào thư mục Downloads của máy...');
            try {
                const fd = new FormData();
                fd.append('filename', currentCompressBatchData.zip_filename);
                const res = await fetch('/api/save-compress-zip-to-downloads', {
                    method: 'POST',
                    body: fd
                });
                const d = await res.json();
                if (d.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp file ZIP vào Downloads: ${d.filename}`);
                } else {
                    throw new Error(d.message);
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    if (btnResetCompressBatch) {
        btnResetCompressBatch.addEventListener('click', () => {
            compressBatchResultContainer.classList.add('hidden');
            compressEmptyState.classList.remove('hidden');
            resetCompressInput();
        });
    }

    // Slider Dragging for Compressor Single View
    function setCompressSliderPos(percentage) {
        percentage = Math.max(0, Math.min(100, percentage));
        compressSliderClip.style.width = `${percentage}%`;
        compressSliderHandle.style.left = `${percentage}%`;
    }

    if (compressSliderWrapper) {
        const onDragMove = (e) => {
            if (!isCompressDragging) return;
            const rect = compressSliderWrapper.getBoundingClientRect();
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            if (clientX === undefined) return;
            const offsetX = clientX - rect.left;
            const pct = (offsetX / rect.width) * 100;
            setCompressSliderPos(pct);
        };

        const onDragEnd = () => {
            isCompressDragging = false;
            document.removeEventListener('mousemove', onDragMove);
            document.removeEventListener('mouseup', onDragEnd);
            document.removeEventListener('touchmove', onDragMove);
            document.removeEventListener('touchend', onDragEnd);
        };

        const onDragStart = (e) => {
            isCompressDragging = true;
            document.addEventListener('mousemove', onDragMove);
            document.addEventListener('mouseup', onDragEnd);
            document.addEventListener('touchmove', onDragMove);
            document.addEventListener('touchend', onDragEnd);
            onDragMove(e);
        };

        compressSliderHandle.addEventListener('mousedown', onDragStart);
        compressSliderHandle.addEventListener('touchstart', onDragStart);

        compressSliderWrapper.addEventListener('click', (e) => {
            if (e.target === compressSliderHandle || compressSliderHandle.contains(e.target)) return;
            const rect = compressSliderWrapper.getBoundingClientRect();
            const pct = ((e.clientX - rect.left) / rect.width) * 100;
            setCompressSliderPos(pct);
        });
    }

    // View Mode Tabs for Compressor
    compressViewTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            compressViewTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const mode = tab.getAttribute('data-compress-mode');

            if (mode === 'slider') {
                compressSliderClip.style.display = 'block';
                compressSliderHandle.style.display = 'flex';
                setCompressSliderPos(50);
            } else if (mode === 'compressed') {
                compressSliderClip.style.display = 'block';
                compressSliderHandle.style.display = 'none';
                setCompressSliderPos(0);
            } else if (mode === 'original') {
                compressSliderClip.style.display = 'block';
                compressSliderHandle.style.display = 'none';
                setCompressSliderPos(100);
            }
        });
    });

    // Zoom Controls for Compressor
    if (btnCompressZoomToggle) {
        btnCompressZoomToggle.addEventListener('click', () => {
            isCompressZoomed = !isCompressZoomed;
            if (isCompressZoomed) {
                compressImgResult.style.transform = 'scale(1.5)';
                compressImgOriginal.style.transform = 'scale(1.5)';
                btnCompressZoomToggle.classList.add('active');
                showToast('Đang phóng to 150% để soi cận cảnh chi tiết.');
            } else {
                compressImgResult.style.transform = 'scale(1)';
                compressImgOriginal.style.transform = 'scale(1)';
                btnCompressZoomToggle.classList.remove('active');
            }
        });
    }

    if (btnCompressZoomFit) {
        btnCompressZoomFit.addEventListener('click', () => {
            isCompressZoomed = false;
            compressImgResult.style.transform = 'scale(1)';
            compressImgOriginal.style.transform = 'scale(1)';
            btnCompressZoomToggle.classList.remove('active');
        });
    }

    // Single Download Actions with Original Filename
    if (btnDownloadCompressed) {
        btnDownloadCompressed.addEventListener('click', () => {
            if (!currentSingleCompressItem) return;
            triggerDownload(currentSingleCompressItem.download_url, currentSingleCompressItem.filename);
        });
    }

    if (btnSaveCompressToDownloads) {
        btnSaveCompressToDownloads.addEventListener('click', async () => {
            if (!currentSingleCompressItem) return;
            showToast(`Đang lưu ảnh ${currentSingleCompressItem.filename} vào Downloads...`);
            try {
                const formData = new FormData();
                formData.append('batch_id', currentSingleCompressItem.batch_id || (currentCompressBatchData && currentCompressBatchData.batch_id));
                formData.append('filename', currentSingleCompressItem.filename);
                const res = await fetch('/api/save-compress-item-to-downloads', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp vào Downloads: ${data.filename}`);
                } else {
                    throw new Error(data.message);
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    if (btnResetCompress) {
        btnResetCompress.addEventListener('click', () => {
            compressResultContainer.classList.add('hidden');
            compressEmptyState.classList.remove('hidden');
            resetCompressInput();
        });
    }

    // ==========================================
    // 13. AI IMAGE GENERATOR (T2I & I2I) CLIENT LOGIC
    // ==========================================
    let currentGenMode = 't2i';
    let currentGenFile = null;
    let currentGenSampleId = null;
    let currentGenSourceFilename = null;
    let currentGenResult = null;
    let genTimerInterval = null;
    let isGenProcessing = false;

    // DOM Elements for Generator
    const btnModeT2I = document.getElementById('btnModeT2I');
    const btnModeI2I = document.getElementById('btnModeI2I');
    const genI2iSection = document.getElementById('genI2iSection');
    const genPromptStepBadge = document.getElementById('genPromptStepBadge');
    const genSettingsStepBadge = document.getElementById('genSettingsStepBadge');

    // I2I Upload
    const genDropZone = document.getElementById('genDropZone');
    const genFileInput = document.getElementById('genFileInput');
    const genDropPrompt = document.getElementById('genDropPrompt');
    const genUploadPreviewWrapper = document.getElementById('genUploadPreviewWrapper');
    const genThumb = document.getElementById('genThumb');
    const genFileName = document.getElementById('genFileName');
    const genFileDims = document.getElementById('genFileDims');
    const btnRemoveGenFile = document.getElementById('btnRemoveGenFile');
    const genReuseWrapper = document.getElementById('genReuseWrapper');
    const btnReuseEnhancedForI2i = document.getElementById('btnReuseEnhancedForI2i');
    const genSampleBtns = document.querySelectorAll('.gen-sample-btn');

    // Strength & Sliders
    const genStrength = document.getElementById('genStrength');
    const genStrengthVal = document.getElementById('genStrengthVal');
    const genStrengthHint = document.getElementById('genStrengthHint');

    // Prompt & Presets
    const genPrompt = document.getElementById('genPrompt');
    const btnRandomPrompt = document.getElementById('btnRandomPrompt');
    const btnClearPrompt = document.getElementById('btnClearPrompt');
    const promptChips = document.querySelectorAll('.prompt-chip');
    const genStylePreset = document.getElementById('genStylePreset');
    const genNegativePrompt = document.getElementById('genNegativePrompt');

    // Settings
    const genModelSelect = document.getElementById('genModelSelect');
    const genRatioRadios = document.querySelectorAll('input[name="genAspectRatio"]');
    const genRatioPills = document.querySelectorAll('.ratio-pill');
    const genSteps = document.getElementById('genSteps');
    const genStepsVal = document.getElementById('genStepsVal');
    const genCfg = document.getElementById('genCfg');
    const genCfgVal = document.getElementById('genCfgVal');
    const genSeed = document.getElementById('genSeed');

    // Action & States
    const btnGenerate = document.getElementById('btnGenerate');
    const btnGenerateText = document.getElementById('btnGenerateText');
    const genSpinner = document.getElementById('genSpinner');
    const genBtnIcon = document.getElementById('genBtnIcon');

    const genEmptyState = document.getElementById('genEmptyState');
    const genProcessingState = document.getElementById('genProcessingState');
    const genResultContainer = document.getElementById('genResultContainer');
    const genProcessingStatusText = document.getElementById('genProcessingStatusText');
    const genProcessingSubText = document.getElementById('genProcessingSubText');
    const genProcessingTimer = document.getElementById('genProcessingTimer');

    // Result & Workflow
    const genResultImg = document.getElementById('genResultImg');
    const genResultModeTag = document.getElementById('genResultModeTag');
    const genResultDimsTag = document.getElementById('genResultDimsTag');
    const genStatResolution = document.getElementById('genStatResolution');
    const genStatFilesize = document.getElementById('genStatFilesize');
    const genStatDuration = document.getElementById('genStatDuration');
    const genStatSeed = document.getElementById('genStatSeed');

    const btnGenZoomToggle = document.getElementById('btnGenZoomToggle');
    const btnGenReset = document.getElementById('btnGenReset');
    const btnDownloadGenPng = document.getElementById('btnDownloadGenPng');
    const btnDownloadGenJpg = document.getElementById('btnDownloadGenJpg');
    const btnSaveGenToDownloads = document.getElementById('btnSaveGenToDownloads');

    const btnSendToEnhancer = document.getElementById('btnSendToEnhancer');
    const btnSendToAspect = document.getElementById('btnSendToAspect');
    const btnSendToCompressor = document.getElementById('btnSendToCompressor');
    const btnUseForNextI2i = document.getElementById('btnUseForNextI2i');

    // 1. Mode Switcher (T2I vs I2I)
    function setGenMode(mode) {
        currentGenMode = mode;
        if (btnModeT2I) btnModeT2I.classList.toggle('active', mode === 't2i');
        if (btnModeI2I) btnModeI2I.classList.toggle('active', mode === 'i2i');

        if (genI2iSection) {
            genI2iSection.classList.toggle('hidden', mode !== 'i2i');
        }

        if (mode === 'i2i') {
            if (genPromptStepBadge) genPromptStepBadge.textContent = '3';
            if (genSettingsStepBadge) genSettingsStepBadge.textContent = '4';
            if (btnGenerateText) btnGenerateText.textContent = 'BẮT ĐẦU BIẾN ĐỔI ẢNH (I2I · GPU RTX 3060)';
        } else {
            if (genPromptStepBadge) genPromptStepBadge.textContent = '2';
            if (genSettingsStepBadge) genSettingsStepBadge.textContent = '3';
            if (btnGenerateText) btnGenerateText.textContent = 'BẮT ĐẦU TẠO ẢNH AI (T2I · GPU RTX 3060)';
        }
    }

    if (btnModeT2I) btnModeT2I.addEventListener('click', () => setGenMode('t2i'));
    if (btnModeI2I) btnModeI2I.addEventListener('click', () => setGenMode('i2i'));

    // 2. I2I File Handling
    function handleGenFileSelect(file) {
        if (!file || !file.type.match('image.*')) {
            showToast('Vui lòng chọn tệp hình ảnh hợp lệ (PNG, JPG, WEBP)!', 'error');
            return;
        }

        currentGenFile = file;
        currentGenSampleId = null;
        currentGenSourceFilename = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                if (genThumb) genThumb.src = e.target.result;
                if (genFileName) genFileName.textContent = file.name;
                const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
                if (genFileDims) genFileDims.textContent = `${img.naturalWidth} x ${img.naturalHeight} · ${sizeMb} MB`;

                if (genDropPrompt) genDropPrompt.classList.add('hidden');
                if (genUploadPreviewWrapper) genUploadPreviewWrapper.classList.remove('hidden');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function resetGenFileInput() {
        currentGenFile = null;
        currentGenSampleId = null;
        currentGenSourceFilename = null;
        if (genFileInput) genFileInput.value = '';
        if (genUploadPreviewWrapper) genUploadPreviewWrapper.classList.add('hidden');
        if (genDropPrompt) genDropPrompt.classList.remove('hidden');
    }

    if (genDropZone) {
        genDropZone.addEventListener('click', (e) => {
            if (e.target !== btnRemoveGenFile && !btnRemoveGenFile?.contains(e.target)) {
                genFileInput?.click();
            }
        });

        ['dragenter', 'dragover'].forEach(name => {
            genDropZone.addEventListener(name, (e) => {
                e.preventDefault();
                genDropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            genDropZone.addEventListener(name, (e) => {
                e.preventDefault();
                genDropZone.classList.remove('dragover');
            });
        });

        genDropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer?.files?.[0]) {
                handleGenFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (genFileInput) {
        genFileInput.addEventListener('change', (e) => {
            if (e.target.files?.[0]) {
                handleGenFileSelect(e.target.files[0]);
            }
        });
    }

    if (btnRemoveGenFile) {
        btnRemoveGenFile.addEventListener('click', (e) => {
            e.stopPropagation();
            resetGenFileInput();
        });
    }

    // I2I Sample Buttons
    genSampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleId = btn.dataset.sample;
            currentGenSampleId = sampleId;
            currentGenFile = null;
            currentGenSourceFilename = null;

            const sampleImgUrl = sampleId === 'anime' ? '/static/samples/sample_anime_lowres.jpg' : '/static/samples/sample_portrait_lowres.jpg';
            if (genThumb) genThumb.src = sampleImgUrl;
            if (genFileName) genFileName.textContent = `Ảnh mẫu (${sampleId})`;
            if (genFileDims) genFileDims.textContent = 'Mẫu thử chuẩn độ phân giải AI';

            if (genDropPrompt) genDropPrompt.classList.add('hidden');
            if (genUploadPreviewWrapper) genUploadPreviewWrapper.classList.remove('hidden');
            showToast(`Đã chọn ảnh mẫu ${sampleId} cho I2I`);
        });
    });

    // Reuse enhanced button
    if (btnReuseEnhancedForI2i) {
        btnReuseEnhancedForI2i.addEventListener('click', () => {
            if (!currentResultData) {
                showToast('Chưa có ảnh nào được làm nét gần đây!', 'error');
                return;
            }
            currentGenSourceFilename = currentResultData.filename;
            currentGenFile = null;
            currentGenSampleId = null;

            if (genThumb) genThumb.src = currentResultData.enhanced_url;
            if (genFileName) genFileName.textContent = currentResultData.filename;
            if (genFileDims) genFileDims.textContent = `${currentResultData.stats?.final_width} x ${currentResultData.stats?.final_height} · ${currentResultData.file_sizes?.enhanced_mb || 0} MB`;

            if (genDropPrompt) genDropPrompt.classList.add('hidden');
            if (genUploadPreviewWrapper) genUploadPreviewWrapper.classList.remove('hidden');
            showToast('Đã nạp ảnh vừa làm nét làm nguồn I2I');
        });
    }

    // Strength Slider & Hint
    if (genStrength) {
        genStrength.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (genStrengthVal) genStrengthVal.textContent = val.toFixed(2);
            if (genStrengthHint) {
                if (val <= 0.35) {
                    genStrengthHint.innerHTML = `<b>${val.toFixed(2)} (Bảo toàn cao)</b>: Giữ tối đa đường nét, phong cách và màu sắc ảnh gốc, chỉ tinh chỉnh nhẹ.`;
                } else if (val <= 0.55) {
                    genStrengthHint.innerHTML = `<b>${val.toFixed(2)} (Cân bằng - Khuyên dùng)</b>: Giữ trọn vẹn nét vẽ & phong cách ảnh gốc, bổ sung chi tiết theo Prompt.`;
                } else {
                    genStrengthHint.innerHTML = `<b>${val.toFixed(2)} (Biến đổi mạnh)</b>: Chuyển đổi sang phong cách mới, dựa trên hình khối ảnh nguồn.`;
                }
            }
        });
    }

    // 3. Prompt Inspiration & Randomizer
    if (btnRandomPrompt) {
        btnRandomPrompt.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/generator-random-prompt');
                const data = await res.json();
                if (data.status === 'success' && data.data) {
                    const item = data.data;
                    if (genPrompt) genPrompt.value = item.prompt;
                    if (genStylePreset && item.style) genStylePreset.value = item.style;
                    if (item.ratio) {
                        const targetRadio = document.querySelector(`input[name="genAspectRatio"][value="${item.ratio}"]`);
                        if (targetRadio) {
                            targetRadio.checked = true;
                            genRatioPills.forEach(p => p.classList.remove('active'));
                            targetRadio.closest('.ratio-pill')?.classList.add('active');
                        }
                    }
                    showToast('🎲 Đã nạp gợi ý Prompt độc đáo!');
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    if (btnClearPrompt) {
        btnClearPrompt.addEventListener('click', () => {
            if (genPrompt) genPrompt.value = '';
            genPrompt?.focus();
        });
    }

    promptChips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (genPrompt) genPrompt.value = chip.dataset.prompt || '';
            if (genStylePreset && chip.dataset.style) genStylePreset.value = chip.dataset.style;
            showToast(`Đã chọn prompt: ${chip.textContent}`);
        });
    });

    // Ratio Pills
    genRatioRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            genRatioPills.forEach(p => p.classList.remove('active'));
            radio.closest('.ratio-pill')?.classList.add('active');
        });
    });

    // Sliders
    if (genSteps) {
        genSteps.addEventListener('input', (e) => {
            if (genStepsVal) genStepsVal.textContent = e.target.value;
        });
    }
    if (genCfg) {
        genCfg.addEventListener('input', (e) => {
            if (genCfgVal) genCfgVal.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }

    // Zoom Toggle
    let isGenZoomed = false;
    if (btnGenZoomToggle && genResultImg) {
        btnGenZoomToggle.addEventListener('click', () => {
            isGenZoomed = !isGenZoomed;
            genResultImg.classList.toggle('zoomed', isGenZoomed);
            btnGenZoomToggle.classList.toggle('active', isGenZoomed);
        });

        genResultImg.addEventListener('click', () => {
            isGenZoomed = !isGenZoomed;
            genResultImg.classList.toggle('zoomed', isGenZoomed);
            btnGenZoomToggle.classList.toggle('active', isGenZoomed);
        });
    }

    if (btnGenReset) {
        btnGenReset.addEventListener('click', () => {
            if (genResultContainer) genResultContainer.classList.add('hidden');
            if (genEmptyState) genEmptyState.classList.remove('hidden');
        });
    }

    // 4. Generate Button Execution
    if (btnGenerate) {
        btnGenerate.addEventListener('click', async () => {
            if (isGenProcessing) return;

            const promptText = genPrompt?.value?.trim() || '';
            if (!promptText) {
                showToast('Vui lòng nhập mô tả ảnh (Prompt) để bắt đầu tạo!', 'error');
                genPrompt?.focus();
                return;
            }

            if (currentGenMode === 'i2i' && !currentGenFile && !currentGenSampleId && !currentGenSourceFilename) {
                showToast('Vui lòng tải lên một ảnh nguồn hoặc chọn ảnh mẫu cho chế độ Image-to-Image!', 'error');
                return;
            }

            // Prepare Form Data
            const formData = new FormData();
            formData.append('mode', currentGenMode);
            formData.append('prompt', promptText);
            formData.append('negative_prompt', genNegativePrompt?.value?.trim() || '');
            formData.append('style_preset', genStylePreset?.value || 'auto');
            formData.append('model_id', genModelSelect?.value || 'runwayml/stable-diffusion-v1-5');

            const selectedRatio = document.querySelector('input[name="genAspectRatio"]:checked')?.value || '1:1';
            formData.append('aspect_ratio', selectedRatio);
            formData.append('steps', genSteps?.value || 25);
            formData.append('cfg_scale', genCfg?.value || 7.5);
            formData.append('seed', genSeed?.value || -1);

            if (currentGenMode === 'i2i') {
                formData.append('strength', genStrength?.value || 0.45);
                if (currentGenFile) {
                    formData.append('image', currentGenFile);
                } else if (currentGenSourceFilename) {
                    formData.append('source_filename', currentGenSourceFilename);
                } else if (currentGenSampleId) {
                    formData.append('sample_id', currentGenSampleId);
                }
            }

            // Set UI to processing
            isGenProcessing = true;
            btnGenerate.disabled = true;
            if (genSpinner) genSpinner.classList.remove('hidden');
            if (genBtnIcon) genBtnIcon.classList.add('hidden');
            if (btnGenerateText) btnGenerateText.textContent = 'ĐANG TÍNH TOÁN DIFFUSION TRÊN GPU...';

            if (genEmptyState) genEmptyState.classList.add('hidden');
            if (genResultContainer) genResultContainer.classList.add('hidden');
            if (genProcessingState) genProcessingState.classList.remove('hidden');

            const tStart = performance.now();
            if (genProcessingTimer) genProcessingTimer.textContent = '0.0s';
            if (genTimerInterval) clearInterval(genTimerInterval);
            genTimerInterval = setInterval(() => {
                const elapsed = ((performance.now() - tStart) / 1000).toFixed(1);
                if (genProcessingTimer) genProcessingTimer.textContent = `${elapsed}s`;
            }, 100);

            try {
                const res = await fetch('/api/generate-image', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.status !== 'success') {
                    throw new Error(data.message || 'Lỗi sinh ảnh từ server');
                }

                currentGenResult = data.data;
                const meta = currentGenResult.meta || {};

                // Update Result UI
                if (genResultImg) genResultImg.src = currentGenResult.result_url;
                if (genResultModeTag) genResultModeTag.textContent = (currentGenResult.mode || 'T2I').toUpperCase();
                if (genResultDimsTag) genResultDimsTag.textContent = `${meta.width || 512} x ${meta.height || 512}`;

                if (genStatResolution) genStatResolution.textContent = `${meta.width || 512} x ${meta.height || 512}`;
                if (genStatFilesize) genStatFilesize.textContent = `${currentGenResult.file_sizes?.result_kb || 0} KB`;
                if (genStatDuration) genStatDuration.textContent = `${meta.duration_sec || 0}s`;
                if (genStatSeed) genStatSeed.textContent = meta.seed || 'Ngẫu nhiên';

                if (btnDownloadGenPng) {
                    btnDownloadGenPng.href = currentGenResult.download_url;
                    btnDownloadGenPng.download = `ClarityAI_${currentGenResult.filename}`;
                }
                if (btnDownloadGenJpg) {
                    btnDownloadGenJpg.href = currentGenResult.download_jpg_url;
                    btnDownloadGenJpg.download = `ClarityAI_${currentGenResult.filename.replace('.png', '.jpg')}`;
                }

                // Show Result Container
                if (genProcessingState) genProcessingState.classList.add('hidden');
                if (genResultContainer) genResultContainer.classList.remove('hidden');

                // Refresh history drawer
                fetchHistory();

                showToast(`✨ Tạo ảnh thành công trong ${meta.duration_sec}s trên GPU RTX 3060!`);

            } catch (err) {
                console.error(err);
                showToast(`Lỗi tạo ảnh: ${err.message}`, 'error');
                if (genProcessingState) genProcessingState.classList.add('hidden');
                if (genEmptyState) genEmptyState.classList.remove('hidden');
            } finally {
                isGenProcessing = false;
                btnGenerate.disabled = false;
                if (genSpinner) genSpinner.classList.add('hidden');
                if (genBtnIcon) genBtnIcon.classList.remove('hidden');
                if (btnGenerateText) {
                    btnGenerateText.textContent = currentGenMode === 'i2i'
                        ? 'BẮT ĐẦU BIẾN ĐỔI ẢNH (I2I · GPU RTX 3060)'
                        : 'BẮT ĐẦU TẠO ẢNH AI (T2I · GPU RTX 3060)';
                }
                if (genTimerInterval) clearInterval(genTimerInterval);
            }
        });
    }

    // 5. Workflow Inter-tab Integration
    // Save to Downloads directly
    if (btnSaveGenToDownloads) {
        btnSaveGenToDownloads.addEventListener('click', async () => {
            if (!currentGenResult?.filename) return;
            showToast(`Đang lưu ảnh ${currentGenResult.filename} vào Downloads...`);
            try {
                const formData = new FormData();
                formData.append('filename', currentGenResult.filename);
                const res = await fetch('/api/save-to-downloads', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã lưu trực tiếp vào Downloads: ${data.filename}`);
                } else {
                    throw new Error(data.message);
                }
            } catch (err) {
                showToast(`Lỗi: ${err.message}`, 'error');
            }
        });
    }

    // Send to Enhancer (Upscale 2x/4x)
    if (btnSendToEnhancer) {
        btnSendToEnhancer.addEventListener('click', () => {
            if (!currentGenResult?.filename) return;

            // Switch to Enhancer Tab
            switchAppTab('enhancer');

            // Load generated image into Enhancer Preview
            currentFile = null;
            currentSampleId = null;

            if (uploadThumb) uploadThumb.src = currentGenResult.result_url;
            if (previewFileName) previewFileName.textContent = currentGenResult.filename;
            if (previewFileDims) previewFileDims.textContent = `${currentGenResult.meta?.width || 512} x ${currentGenResult.meta?.height || 512} · ${currentGenResult.file_sizes?.result_kb || 0} KB`;

            if (dropPrompt) dropPrompt.classList.add('hidden');
            if (uploadPreviewWrapper) uploadPreviewWrapper.classList.remove('hidden');

            showToast('🔍 Đã chuyển ảnh sang Siêu Làm Nét! Chọn mức 2x hoặc 4x và bấm "Bắt Đầu Làm Nét".');
        });
    }

    // Send to Aspect Converter
    if (btnSendToAspect) {
        btnSendToAspect.addEventListener('click', () => {
            if (!currentGenResult?.filename) return;

            // Switch to Aspect Tab
            switchAppTab('aspect');

            // Set up aspect source
            currentAspectSourceFilename = currentGenResult.filename;
            currentAspectFile = null;
            currentAspectSampleId = null;

            if (aspectThumb) aspectThumb.src = currentGenResult.result_url;
            if (aspectFileName) aspectFileName.textContent = currentGenResult.filename;
            if (aspectFileDims) aspectFileDims.textContent = `${currentGenResult.meta?.width || 512} x ${currentGenResult.meta?.height || 512} · ${currentGenResult.file_sizes?.result_kb || 0} KB`;

            if (aspectDropPrompt) aspectDropPrompt.classList.add('hidden');
            if (aspectPreviewWrapper) aspectPreviewWrapper.classList.remove('hidden');

            showToast('📐 Đã chuyển ảnh sang Đổi Tỷ Lệ! Chọn tỷ lệ 9:16 hoặc 16:9 và bấm chuyển đổi.');
        });
    }

    // Send to Compressor
    if (btnSendToCompressor) {
        btnSendToCompressor.addEventListener('click', () => {
            if (!currentGenResult?.filename) return;
            switchAppTab('compress');
            showToast('🗜️ Đã chuyển sang Nén Dung Lượng!');
        });
    }

    // Use for next I2I iteration
    if (btnUseForNextI2i) {
        btnUseForNextI2i.addEventListener('click', () => {
            if (!currentGenResult?.filename) return;

            setGenMode('i2i');
            currentGenSourceFilename = currentGenResult.filename;
            currentGenFile = null;
            currentGenSampleId = null;

            if (genThumb) genThumb.src = currentGenResult.result_url;
            if (genFileName) genFileName.textContent = currentGenResult.filename;
            if (genFileDims) genFileDims.textContent = `${currentGenResult.meta?.width || 512} x ${currentGenResult.meta?.height || 512} · ${currentGenResult.file_sizes?.result_kb || 0} KB`;

            if (genDropPrompt) genDropPrompt.classList.add('hidden');
            if (genUploadPreviewWrapper) genUploadPreviewWrapper.classList.remove('hidden');

            showToast('🔄 Đã nạp ảnh làm nguồn cho vòng sáng tạo Image-to-Image tiếp theo!');
        });
    }

    // Clipboard paste (Ctrl + V) when on Generator tab
    window.addEventListener('paste', (e) => {
        if (!viewGenerator || viewGenerator.classList.contains('hidden')) return;
        if (currentGenMode !== 'i2i') return;

        const items = e.clipboardData?.items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) {
                    handleGenFileSelect(file);
                    showToast('📋 Đã dán ảnh từ Clipboard vào Image-to-Image!');
                    break;
                }
            }
        }
    });

    // ========================================================================
    // 12. INLINE AI PROMPT OPTIMIZER (GENERATOR STUDIO)
    // ========================================================================
    const optNaturalInput = document.getElementById('optNaturalInput');
    const btnRunPromptOpt = document.getElementById('btnRunPromptOpt');
    const aiOptResultPreview = document.getElementById('aiOptResultPreview');
    const aiOptStyleTag = document.getElementById('aiOptStyleTag');
    const aiOptPreviewText = document.getElementById('aiOptPreviewText');
    const btnApplyOptimizedPrompt = document.getElementById('btnApplyOptimizedPrompt');

    let currentOptimizedPromptData = null;

    if (btnRunPromptOpt && optNaturalInput) {
        btnRunPromptOpt.addEventListener('click', runInlinePromptOptimizer);
        optNaturalInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                runInlinePromptOptimizer();
            }
        });
    }

    async function runInlinePromptOptimizer() {
        const text = optNaturalInput.value.trim();
        if (!text) {
            showToast('Vui lòng nhập ý tưởng hoặc câu mô tả tự nhiên!', 'error');
            optNaturalInput.focus();
            return;
        }

        const originalBtnHtml = btnRunPromptOpt.innerHTML;
        btnRunPromptOpt.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tối ưu...';
        btnRunPromptOpt.disabled = true;

        try {
            const currentStyle = genStylePreset ? genStylePreset.value : 'auto';
            const res = await fetch('/api/llm/optimize-prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    style_preset: currentStyle
                })
            });

            const data = await res.json();
            if (data.status === 'success' && data.data) {
                currentOptimizedPromptData = data.data;
                aiOptPreviewText.textContent = data.data.optimized_prompt;
                aiOptStyleTag.textContent = data.data.style_preset ? `Phong cách: ${data.data.style_preset}` : 'Điện ảnh 8K';
                aiOptResultPreview.classList.remove('hidden');
                showToast(`⚡ AI đã tối ưu Prompt xong trong ${data.data.duration_sec || 0.05}s!`);
            } else {
                throw new Error(data.message || 'Lỗi khi tối ưu prompt');
            }
        } catch (err) {
            console.error('Prompt Optimizer Error:', err);
            showToast(`Lỗi: ${err.message}`, 'error');
        } finally {
            btnRunPromptOpt.innerHTML = originalBtnHtml;
            btnRunPromptOpt.disabled = false;
        }
    }

    if (btnApplyOptimizedPrompt) {
        btnApplyOptimizedPrompt.addEventListener('click', () => {
            if (!currentOptimizedPromptData) return;

            if (genPrompt) {
                genPrompt.value = currentOptimizedPromptData.optimized_prompt;
                genPrompt.focus();
            }
            if (genNegativePrompt && currentOptimizedPromptData.negative_prompt) {
                genNegativePrompt.value = currentOptimizedPromptData.negative_prompt;
            }
            if (genStylePreset && currentOptimizedPromptData.style_preset) {
                genStylePreset.value = currentOptimizedPromptData.style_preset;
            }

            aiOptResultPreview.classList.add('hidden');
            showToast('✅ Đã nạp Prompt chuẩn 8K & Negative Prompt vào Studio Tạo Ảnh!');
        });
    }

    // ========================================================================
    // 13. TRỢ LÝ AI & CHATBOT CONSULTANT (VIEW 6)
    // ========================================================================
    const chatMessagesContainer = document.getElementById('chatMessagesContainer');
    const chatInputText = document.getElementById('chatInputText');
    const btnSendChatMessage = document.getElementById('btnSendChatMessage');
    const chatFileInput = document.getElementById('chatFileInput');
    const btnAttachChatImage = document.getElementById('btnAttachChatImage');
    const chatAttachedImageBar = document.getElementById('chatAttachedImageBar');
    const chatAttachedThumb = document.getElementById('chatAttachedThumb');
    const chatAttachedName = document.getElementById('chatAttachedName');
    const chatAttachedSize = document.getElementById('chatAttachedSize');
    const btnRemoveAttachedImg = document.getElementById('btnRemoveAttachedImg');
    const chatDropTarget = document.getElementById('chatDropTarget');
    const btnClearChatHistory = document.getElementById('btnClearChatHistory');

    // LLM Provider Hub DOM
    const llmProviderSelect = document.getElementById('llmProviderSelect');
    const llmStatusDot = document.getElementById('llmStatusDot');
    const llmStatusTitle = document.getElementById('llmStatusTitle');
    const llmProviderDesc = document.getElementById('llmProviderDesc');
    const llmExternalConfig = document.getElementById('llmExternalConfig');
    const llmModelSelect = document.getElementById('llmModelSelect');
    const btnTestLLMConnection = document.getElementById('btnTestLLMConnection');
    const llmPingResult = document.getElementById('llmPingResult');
    const chatActiveEngineText = document.getElementById('chatActiveEngineText');

    let currentChatAttachedFile = null;
    let chatConversationHistory = [];
    let isChatProcessing = false;

    // Fetch LLM Status on Load
    fetchLLMProvidersStatus();

    async function fetchLLMProvidersStatus() {
        try {
            const res = await fetch('/api/llm/status');
            const data = await res.json();
            if (data.status === 'success' && data.data) {
                renderLLMStatus(data.data);
            }
        } catch (e) {
            console.warn('LLM Status check failed', e);
        }
    }

    function renderLLMStatus(statusData) {
        const activeProv = statusData.active_provider || 'built_in';
        if (llmProviderSelect) llmProviderSelect.value = activeProv;

        const provInfo = statusData.providers?.[activeProv];
        const isOnline = provInfo?.available !== false;

        // Update dot
        if (llmStatusDot) {
            llmStatusDot.className = isOnline ? 'pulse-indicator-green' : 'pulse-indicator-red';
        }

        // Update badge
        const badgeEl = document.querySelector('.badge-free');
        if (badgeEl) {
            badgeEl.textContent = provInfo?.badge || (isOnline ? 'SẴN SÀNG' : 'CHƯA BẬT');
            badgeEl.className = isOnline ? 'badge-free' : 'badge-free badge-offline';
        }

        // Update chat header online dot & subtitle
        const chatOnlineDot = document.querySelector('.chat-online-dot');
        if (chatOnlineDot) {
            chatOnlineDot.className = isOnline ? 'chat-online-dot' : 'chat-online-dot offline';
        }

        if (activeProv === 'built_in') {
            if (llmStatusTitle) llmStatusTitle.textContent = '⚡ Clarity Local Hybrid Engine';
            if (llmProviderDesc) llmProviderDesc.textContent = 'Mô hình chuyên gia cục bộ tích hợp sẵn, phản hồi tức thì 0.05s, 100% Free.';
            if (chatActiveEngineText) chatActiveEngineText.textContent = 'Clarity AI Hybrid Engine (Local & Sẵn sàng)';
            if (llmExternalConfig) llmExternalConfig.classList.add('hidden');
        } else if (activeProv === 'local_qwen') {
            if (llmStatusTitle) llmStatusTitle.textContent = '🤖 Qwen 2.5 Instruct Local GPU';
            if (llmProviderDesc) llmProviderDesc.textContent = 'Mô hình ngôn ngữ lớn Qwen 2.5 Instruct chạy trực tiếp trên GPU RTX 3060 của máy, hiểu tiếng Việt và tối ưu prompt sáng tạo xuất sắc.';
            if (chatActiveEngineText) chatActiveEngineText.textContent = 'Qwen 2.5 Instruct (Local RTX 3060 CUDA)';
            if (llmExternalConfig) llmExternalConfig.classList.add('hidden');
        } else if (activeProv === 'ollama') {
            const ollamaInfo = statusData.providers?.ollama;
            if (llmStatusTitle) llmStatusTitle.textContent = '🦙 Ollama Local Server';
            if (isOnline) {
                if (llmProviderDesc) llmProviderDesc.textContent = `Đã kết nối với Ollama (${ollamaInfo?.models?.length || 0} model). Sẵn sàng phục vụ.`;
                if (chatActiveEngineText) chatActiveEngineText.textContent = `Ollama (${ollamaInfo?.current_model || 'Local'})`;
            } else {
                if (llmProviderDesc) llmProviderDesc.textContent = '⚠️ Chưa bật dịch vụ Ollama tại localhost:11434. Đang tự động sử dụng Clarity Local Engine làm dự phòng.';
                if (chatActiveEngineText) chatActiveEngineText.textContent = 'Ollama (Chưa bật ➔ Đang dùng Dự phòng)';
            }
            if (llmExternalConfig) llmExternalConfig.classList.remove('hidden');
            populateModelDropdown(ollamaInfo?.models || []);
        } else if (activeProv === 'lm_studio') {
            const lmInfo = statusData.providers?.lm_studio;
            if (llmStatusTitle) llmStatusTitle.textContent = '🖥️ LM Studio / Local OpenAI API';
            if (isOnline) {
                if (llmProviderDesc) llmProviderDesc.textContent = `Đã kết nối với LM Studio tại ${lmInfo?.host}. Sẵn sàng phục vụ.`;
                if (chatActiveEngineText) chatActiveEngineText.textContent = `LM Studio (${lmInfo?.current_model || 'Local'})`;
            } else {
                if (llmProviderDesc) llmProviderDesc.textContent = '⚠️ Chưa kết nối LM Studio tại localhost:1234. Đang tự động sử dụng Clarity Local Engine làm dự phòng.';
                if (chatActiveEngineText) chatActiveEngineText.textContent = 'LM Studio (Chưa kết nối ➔ Đang dùng Dự phòng)';
            }
            if (llmExternalConfig) llmExternalConfig.classList.remove('hidden');
            populateModelDropdown(lmInfo?.models || []);
        }
    }

    function populateModelDropdown(models) {
        if (!llmModelSelect) return;
        llmModelSelect.innerHTML = '';
        if (models.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Chưa phát hiện model (Chạy "ollama run llama3.2")';
            llmModelSelect.appendChild(opt);
            return;
        }
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            llmModelSelect.appendChild(opt);
        });
    }

    if (llmProviderSelect) {
        llmProviderSelect.addEventListener('change', async (e) => {
            const selected = e.target.value;
            showToast(`Đang chuyển sang nguồn AI: ${selected}...`);
            try {
                const res = await fetch('/api/llm/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider: selected })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`✅ Đã kích hoạt nguồn AI: ${selected.toUpperCase()}`);
                    fetchLLMProvidersStatus();
                }
            } catch (err) {
                showToast(`Lỗi chuyển nguồn AI: ${err.message}`, 'error');
            }
        });
    }

    if (btnTestLLMConnection) {
        btnTestLLMConnection.addEventListener('click', async () => {
            llmPingResult.textContent = 'Đang kiểm tra ping...';
            llmPingResult.className = 'llm-ping-result text-muted';
            try {
                const res = await fetch('/api/llm/status');
                const data = await res.json();
                if (data.status === 'success') {
                    const active = data.data.active_provider;
                    const provInfo = data.data.providers?.[active];
                    if (provInfo?.available) {
                        llmPingResult.textContent = `🟢 Kết nối tốt (${provInfo.count || 0} model sẵn sàng)`;
                        llmPingResult.className = 'llm-ping-result highlight-green';
                        showToast(`🟢 Kết nối ${active.toUpperCase()} thành công!`);
                        populateModelDropdown(provInfo.models || []);
                    } else {
                        llmPingResult.textContent = `🔴 Không phản hồi tại ${provInfo?.host}`;
                        llmPingResult.className = 'llm-ping-result text-danger';
                        showToast(`🔴 Không thể kết nối tới ${provInfo?.host}. Hãy khởi động server trước!`, 'error');
                    }
                }
            } catch (err) {
                llmPingResult.textContent = '🔴 Lỗi mạng';
                showToast('Không thể kiểm tra kết nối', 'error');
            }
        });
    }

    // Image Attachment Handlers for Chatbot
    if (btnAttachChatImage && chatFileInput) {
        btnAttachChatImage.addEventListener('click', () => chatFileInput.click());
        chatFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleChatFileAttach(e.target.files[0]);
            }
        });
    }

    function handleChatFileAttach(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Vui lòng chọn tệp hình ảnh!', 'error');
            return;
        }
        currentChatAttachedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            if (chatAttachedThumb) chatAttachedThumb.src = e.target.result;
            if (chatAttachedName) chatAttachedName.textContent = file.name;
            if (chatAttachedSize) chatAttachedSize.textContent = `${Math.round(file.size / 1024)} KB`;
            if (chatAttachedImageBar) chatAttachedImageBar.classList.remove('hidden');
            showToast(`📸 Đã đính kèm ảnh: ${file.name}. Bấm gửi để biến ảnh thành Prompt!`);
        };
        reader.readAsDataURL(file);
    }

    if (btnRemoveAttachedImg) {
        btnRemoveAttachedImg.addEventListener('click', () => {
            currentChatAttachedFile = null;
            if (chatFileInput) chatFileInput.value = '';
            if (chatAttachedImageBar) chatAttachedImageBar.classList.add('hidden');
        });
    }

    // Drag-and-drop on Chat Box
    if (chatDropTarget) {
        ['dragenter', 'dragover'].forEach(name => {
            chatDropTarget.addEventListener(name, (e) => {
                e.preventDefault();
                chatDropTarget.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(name => {
            chatDropTarget.addEventListener(name, (e) => {
                e.preventDefault();
                chatDropTarget.classList.remove('dragover');
            });
        });
        chatDropTarget.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files[0]) {
                handleChatFileAttach(dt.files[0]);
            }
        });
    }

    // Clipboard Paste (Ctrl+V) on Chatbot Tab
    window.addEventListener('paste', (e) => {
        if (!viewChatbot || viewChatbot.classList.contains('hidden')) return;
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) {
                    handleChatFileAttach(file);
                    showToast('📋 Đã dán ảnh từ Clipboard vào khung chat!');
                    break;
                }
            }
        }
    });

    // Chat Input Text Auto-resize & Enter to Send
    if (chatInputText) {
        chatInputText.addEventListener('input', () => {
            chatInputText.style.height = 'auto';
            chatInputText.style.height = Math.min(chatInputText.scrollHeight, 120) + 'px';
        });

        chatInputText.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }

    if (btnSendChatMessage) {
        btnSendChatMessage.addEventListener('click', sendChatMessage);
    }

    // Topic Buttons in Sidebar
    document.querySelectorAll('.chat-topic-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            if (query && chatInputText) {
                chatInputText.value = query;
                sendChatMessage();
            }
        });
    });

    // Clear Chat History
    if (btnClearChatHistory) {
        btnClearChatHistory.addEventListener('click', () => {
            chatConversationHistory = [];
            if (chatMessagesContainer) {
                chatMessagesContainer.innerHTML = `
                    <div class="chat-bubble-row bot">
                        <div class="chat-avatar bot-avatar">
                            <i class="fa-solid fa-sparkles"></i>
                        </div>
                        <div class="chat-bubble bot-bubble">
                            <div class="chat-bubble-content">
                                <p>Đã làm mới cuộc trò chuyện. Hãy hỏi tôi về kỹ thuật nhiếp ảnh, bảng màu hoặc đính kèm ảnh để biến thành prompt!</p>
                            </div>
                        </div>
                    </div>
                `;
            }
            showToast('🧹 Đã xóa lịch sử trò chuyện!');
        });
    }

    // Main Send Message Function
    async function sendChatMessage() {
        if (isChatProcessing) return;

        const text = chatInputText ? chatInputText.value.trim() : '';
        const attachedFile = currentChatAttachedFile;

        if (!text && !attachedFile) {
            showToast('Vui lòng nhập tin nhắn hoặc đính kèm ảnh!', 'error');
            return;
        }

        // Render User Message Bubble
        appendUserMessageBubble(text, attachedFile);

        // Clear input and attached preview
        if (chatInputText) {
            chatInputText.value = '';
            chatInputText.style.height = 'auto';
        }
        currentChatAttachedFile = null;
        if (chatFileInput) chatFileInput.value = '';
        if (chatAttachedImageBar) chatAttachedImageBar.classList.add('hidden');

        // Render Typing Indicator Bubble
        const typingId = appendTypingIndicator();
        isChatProcessing = true;
        if (btnSendChatMessage) btnSendChatMessage.disabled = true;

        try {
            const formData = new FormData();
            formData.append('message', text);
            if (attachedFile) {
                formData.append('image', attachedFile);
            }
            if (chatConversationHistory.length > 0) {
                formData.append('history', JSON.stringify(chatConversationHistory.slice(-4)));
            }

            const res = await fetch('/api/llm/chat', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            removeTypingIndicator(typingId);

            if (data.status === 'success') {
                appendBotMessageBubble(data.reply, data.prompt_data, attachedFile);
                chatConversationHistory.push({ role: 'user', content: text || 'Phân tích hình ảnh này' });
                chatConversationHistory.push({ role: 'assistant', content: data.reply });
            } else {
                throw new Error(data.message || 'Lỗi khi trò chuyện với AI');
            }
        } catch (err) {
            console.error('Chat error:', err);
            removeTypingIndicator(typingId);
            appendBotMessageBubble(`⚠️ **Lỗi:** Không thể kết nối với mô hình AI (${err.message}). Vui lòng thử lại!`);
        } finally {
            isChatProcessing = false;
            if (btnSendChatMessage) btnSendChatMessage.disabled = false;
            if (chatInputText) chatInputText.focus();
        }
    }

    function appendUserMessageBubble(text, file) {
        if (!chatMessagesContainer) return;
        const row = document.createElement('div');
        row.className = 'chat-bubble-row user';

        let imgHtml = '';
        if (file) {
            const previewUrl = URL.createObjectURL(file);
            imgHtml = `<img src="${previewUrl}" alt="Ảnh tải lên" class="user-msg-image-thumb">`;
        }

        const msgContent = text ? `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>` : '';

        row.innerHTML = `
            <div class="chat-avatar user-avatar">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="chat-bubble user-bubble">
                <div class="chat-bubble-content">
                    ${imgHtml}
                    ${msgContent}
                </div>
            </div>
        `;

        chatMessagesContainer.appendChild(row);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    function appendTypingIndicator() {
        if (!chatMessagesContainer) return null;
        const id = 'typing_' + Date.now();
        const row = document.createElement('div');
        row.className = 'chat-bubble-row bot';
        row.id = id;
        row.innerHTML = `
            <div class="chat-avatar bot-avatar">
                <i class="fa-solid fa-wand-magic-sparkles"></i>
            </div>
            <div class="chat-bubble bot-bubble">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatMessagesContainer.appendChild(row);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        if (!id) return;
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendBotMessageBubble(replyMarkdown, promptData, originalFile) {
        if (!chatMessagesContainer) return;

        const row = document.createElement('div');
        row.className = 'chat-bubble-row bot';

        const parsedContent = parseSimpleMarkdown(replyMarkdown);

        let actionsHtml = '';
        if (promptData && promptData.generated_prompt) {
            actionsHtml = `
                <div class="chat-prompt-actions-row">
                    <button type="button" class="btn-chat-cta btn-transfer-t2i" title="Chuyển prompt sang Studio Tạo Ảnh">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Sử Dụng Cho Tạo Ảnh (T2I)
                    </button>
                    <button type="button" class="btn-chat-cta btn-chat-cta-purple btn-transfer-i2i" title="Nạp ảnh và prompt sang Image-to-Image">
                        <i class="fa-solid fa-image"></i> Chuyển Sang Image-to-Image
                    </button>
                    <button type="button" class="btn-chat-cta btn-copy-chat-prompt" title="Sao chép prompt">
                        <i class="fa-regular fa-copy"></i> Sao Chép Prompt
                    </button>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="chat-avatar bot-avatar">
                <i class="fa-solid fa-wand-magic-sparkles"></i>
            </div>
            <div class="chat-bubble bot-bubble">
                <div class="chat-bubble-content">
                    ${parsedContent}
                    ${actionsHtml}
                </div>
            </div>
        `;

        // Wire Action Buttons
        if (promptData) {
            const btnT2i = row.querySelector('.btn-transfer-t2i');
            if (btnT2i) {
                btnT2i.addEventListener('click', () => {
                    switchAppTab('generator');
                    setGenMode('t2i');
                    if (genPrompt) genPrompt.value = promptData.generated_prompt;
                    if (genNegativePrompt && promptData.negative_prompt) genNegativePrompt.value = promptData.negative_prompt;
                    if (genStylePreset && promptData.style_preset) genStylePreset.value = promptData.style_preset;
                    showToast('🎨 Đã nạp Prompt & Phong Cách sang Tạo Ảnh AI (T2I)!');
                    if (genPrompt) genPrompt.focus();
                });
            }

            const btnI2i = row.querySelector('.btn-transfer-i2i');
            if (btnI2i) {
                btnI2i.addEventListener('click', () => {
                    switchAppTab('generator');
                    setGenMode('i2i');
                    if (genPrompt) genPrompt.value = promptData.generated_prompt;
                    if (genNegativePrompt && promptData.negative_prompt) genNegativePrompt.value = promptData.negative_prompt;
                    if (genStylePreset && promptData.style_preset) genStylePreset.value = promptData.style_preset;

                    if (originalFile) {
                        handleGenFileSelect(originalFile);
                    }
                    showToast('🖼️ Đã nạp ảnh nguồn và Prompt sang Image-to-Image (I2I)!');
                    if (genPrompt) genPrompt.focus();
                });
            }

            const btnCopy = row.querySelector('.btn-copy-chat-prompt');
            if (btnCopy) {
                btnCopy.addEventListener('click', () => {
                    navigator.clipboard.writeText(promptData.generated_prompt).then(() => {
                        showToast('📋 Đã sao chép Prompt vào bộ nhớ tạm!');
                    }).catch(() => {
                        showToast('Không thể sao chép tự động', 'error');
                    });
                });
            }
        }

        chatMessagesContainer.appendChild(row);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // Delegated click handler on color swatch chips to copy HEX
    if (chatMessagesContainer) {
        chatMessagesContainer.addEventListener('click', (e) => {
            const chip = e.target.closest('.color-swatch-chip');
            if (chip) {
                const hexMatch = chip.textContent.match(/#[0-9A-Fa-f]{6}/);
                if (hexMatch) {
                    const hex = hexMatch[0];
                    navigator.clipboard.writeText(hex).then(() => {
                        showToast(`🎨 Đã sao chép mã màu: ${hex}!`);
                    });
                }
            }
        });
    }

    // Markdown Parser Helper for Chatbot
    function parseSimpleMarkdown(md) {
        if (!md) return '';

        // Code blocks: ```lang\ncode\n```
        let html = md.replace(/```([\w]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const cleanCode = code.trim();
            return `<div class="code-block-wrap"><button type="button" class="btn-copy-code" onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText); window.showToast('📋 Đã sao chép prompt!');"><i class="fa-regular fa-copy"></i> Sao Chép</button><pre><code class="language-${lang}">${escapeHtml(cleanCode)}</code></pre></div>`;
        });

        // Inline code `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

        // Bold & Italic
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

        // Horizontal rules
        html = html.replace(/^---$/gim, '<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:12px 0;">');

        // Lists
        html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
        html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');

        // Wrap list items
        html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // Paragraphs
        const paragraphs = html.split(/\n{2,}/);
        html = paragraphs.map(p => {
            p = p.trim();
            if (!p) return '';
            if (p.startsWith('<h') || p.startsWith('<pre') || p.startsWith('<ul') || p.startsWith('<blockquote') || p.startsWith('<hr')) {
                return p;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ==========================================
    // 16. BACKGROUND REMOVER CLIENT LOGIC (RMBG-1.4)
    // ==========================================
    const bgRemoverDropZone = document.getElementById('bgRemoverDropZone');
    const bgRemoverFileInput = document.getElementById('bgRemoverFileInput');
    const bgRemoverDropPrompt = document.getElementById('bgRemoverDropPrompt');
    const bgRemoverPreviewWrapper = document.getElementById('bgRemoverPreviewWrapper');
    const bgRemoverThumb = document.getElementById('bgRemoverThumb');
    const bgRemoverFileName = document.getElementById('bgRemoverFileName');
    const bgRemoverFileDims = document.getElementById('bgRemoverFileDims');
    const btnRemoveBgRemoverFile = document.getElementById('btnRemoveBgRemoverFile');
    const bgRemoverFileCountBadge = document.getElementById('bgRemoverFileCountBadge');

    const bgRemoverFilesListWrapper = document.getElementById('bgRemoverFilesListWrapper');
    const bgRemoverFilesGrid = document.getElementById('bgRemoverFilesGrid');
    const bgRemoverSelectedCount = document.getElementById('bgRemoverSelectedCount');
    const btnClearBgRemoverFiles = document.getElementById('btnClearBgRemoverFiles');
    const btnAddMoreBgRemoverFiles = document.getElementById('btnAddMoreBgRemoverFiles');

    const btnReuseForBgRemover = document.getElementById('btnReuseForBgRemover');
    const btnRunBgRemove = document.getElementById('btnRunBgRemove');
    const btnBgRemoveText = document.getElementById('btnBgRemoveText');

    const bgRemoverStatusBadge = document.getElementById('bgRemoverStatusBadge');
    const bgRemoverViewToggles = document.getElementById('bgRemoverViewToggles');
    const bgToggleCompare = document.getElementById('bgToggleCompare');
    const bgToggleResultOnly = document.getElementById('bgToggleResultOnly');
    const bgToggleOriginal = document.getElementById('bgToggleOriginal');

    const bgRemoverEmptyState = document.getElementById('bgRemoverEmptyState');
    const bgRemoverProcessingState = document.getElementById('bgRemoverProcessingState');
    const bgRemoverCompareContainer = document.getElementById('bgRemoverCompareContainer');
    const bgResultOnlyContainer = document.getElementById('bgResultOnlyContainer');
    const bgRemoverResultFooter = document.getElementById('bgRemoverResultFooter');

    const bgOrigImg = document.getElementById('bgOrigImg');
    const bgResultImg = document.getElementById('bgResultImg');
    const bgSingleResultImg = document.getElementById('bgSingleResultImg');
    const bgBeforeWrapper = document.getElementById('bgBeforeWrapper');
    const bgCompareHandle = document.getElementById('bgCompareHandle');

    const bgMetaDims = document.getElementById('bgMetaDims');
    const bgMetaDuration = document.getElementById('bgMetaDuration');
    const bgMetaModel = document.getElementById('bgMetaModel');
    const bgMetaSize = document.getElementById('bgMetaSize');

    const btnDownloadBgPng = document.getElementById('btnDownloadBgPng');
    const btnDownloadBgJpg = document.getElementById('btnDownloadBgJpg');
    const btnSaveBgToDownloads = document.getElementById('btnSaveBgToDownloads');

    const bgBatchResultsContainer = document.getElementById('bgBatchResultsContainer');
    const bgBatchResultsGrid = document.getElementById('bgBatchResultsGrid');
    const bgBatchSuccessCount = document.getElementById('bgBatchSuccessCount');
    const btnDownloadBgZip = document.getElementById('btnDownloadBgZip');

    const bgCustomColorPicker = document.getElementById('bgCustomColorPicker');
    const bgCustomColorCode = document.getElementById('bgCustomColorCode');
    const btnApplyCustomColor = document.getElementById('btnApplyCustomColor');
    const bgFeatherEdge = document.getElementById('bgFeatherEdge');

    const bgModelSelect = document.getElementById('bgModelSelect');
    const bgCleanModeSelect = document.getElementById('bgCleanModeSelect');
    const bgSensitivitySlider = document.getElementById('bgSensitivitySlider');
    const bgSensitivityVal = document.getElementById('bgSensitivityVal');
    const bgDeFringe = document.getElementById('bgDeFringe');

    if (bgSensitivitySlider && bgSensitivityVal) {
        bgSensitivitySlider.addEventListener('input', (e) => {
            bgSensitivityVal.textContent = `${e.target.value}%`;
        });
    }

    let bgRemoverSelectedFiles = [];
    let currentBgResultData = null;
    let selectedRemoverBgColor = 'transparent';

    // Option cards selection
    const bgOptCards = document.querySelectorAll('.bg-opt-card');
    bgOptCards.forEach(card => {
        card.addEventListener('click', () => {
            bgOptCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
            selectedRemoverBgColor = card.dataset.color || 'transparent';
        });
    });

    if (bgCustomColorPicker && bgCustomColorCode) {
        bgCustomColorPicker.addEventListener('input', (e) => {
            bgCustomColorCode.textContent = e.target.value.toUpperCase();
        });
        if (btnApplyCustomColor) {
            btnApplyCustomColor.addEventListener('click', () => {
                selectedRemoverBgColor = bgCustomColorPicker.value;
                bgOptCards.forEach(c => c.classList.remove('active'));
                window.showToast(`🎨 Đã chọn màu phông: ${selectedRemoverBgColor.toUpperCase()}`);
            });
        }
    }

    // File handling
    function setBgRemoverFiles(files) {
        if (!files || files.length === 0) return;
        bgRemoverSelectedFiles = Array.from(files);
        updateBgRemoverUI();
    }

    function updateBgRemoverUI() {
        const count = bgRemoverSelectedFiles.length;
        if (count === 0) {
            bgRemoverPreviewWrapper.classList.add('hidden');
            bgRemoverDropPrompt.classList.remove('hidden');
            bgRemoverFilesListWrapper.classList.add('hidden');
            bgRemoverFileCountBadge.classList.add('hidden');
            btnRunBgRemove.disabled = true;
            btnBgRemoveText.textContent = 'Tách Nền Bằng AI';
            return;
        }

        btnRunBgRemove.disabled = false;
        bgRemoverFileCountBadge.classList.remove('hidden');
        bgRemoverFileCountBadge.textContent = `${count} ảnh`;

        if (count === 1) {
            const file = bgRemoverSelectedFiles[0];
            bgRemoverDropPrompt.classList.add('hidden');
            bgRemoverPreviewWrapper.classList.remove('hidden');
            bgRemoverFilesListWrapper.classList.add('hidden');
            bgRemoverFileName.textContent = file.name;

            const url = URL.createObjectURL(file);
            bgRemoverThumb.src = url;
            const imgTemp = new Image();
            imgTemp.onload = () => {
                const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
                bgRemoverFileDims.textContent = `${imgTemp.naturalWidth} x ${imgTemp.naturalHeight} · ${sizeMb} MB`;
            };
            imgTemp.src = url;
            btnBgRemoveText.textContent = 'Tách Nền Bằng AI (1 Ảnh)';
        } else {
            bgRemoverDropPrompt.classList.add('hidden');
            bgRemoverPreviewWrapper.classList.add('hidden');
            bgRemoverFilesListWrapper.classList.remove('hidden');
            bgRemoverSelectedCount.textContent = count;
            btnBgRemoveText.textContent = `Tách Nền Hàng Loạt (${count} Ảnh)`;

            bgRemoverFilesGrid.innerHTML = '';
            bgRemoverSelectedFiles.forEach((file, idx) => {
                const card = document.createElement('div');
                card.className = 'batch-file-card';
                const url = URL.createObjectURL(file);
                card.innerHTML = `
                    <button type="button" class="btn-remove-batch-item" data-index="${idx}" title="Xóa ảnh này">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                    <img src="${url}" class="batch-file-thumb" alt="thumbnail">
                    <div class="batch-file-name" title="${file.name}">${file.name}</div>
                    <div class="batch-file-meta">
                        <span>#${idx + 1}</span>
                        <span>${(file.size / 1024).toFixed(0)} KB</span>
                    </div>
                `;
                card.querySelector('.btn-remove-batch-item').addEventListener('click', (e) => {
                    e.stopPropagation();
                    bgRemoverSelectedFiles.splice(idx, 1);
                    updateBgRemoverUI();
                });
                bgRemoverFilesGrid.appendChild(card);
            });
        }
    }

    if (bgRemoverDropZone && bgRemoverFileInput) {
        bgRemoverDropZone.addEventListener('click', (e) => {
            if (e.target.closest('.btn-remove-file') || e.target.closest('.btn-remove-batch-item') || e.target.closest('.btn-add-more-files') || e.target.closest('.btn-clear-batch')) return;
            bgRemoverFileInput.click();
        });
        bgRemoverFileInput.addEventListener('change', () => {
            if (bgRemoverFileInput.files.length > 0) {
                setBgRemoverFiles(bgRemoverFileInput.files);
            }
        });

        bgRemoverDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            bgRemoverDropZone.classList.add('drag-active');
        });
        bgRemoverDropZone.addEventListener('dragleave', () => {
            bgRemoverDropZone.classList.remove('drag-active');
        });
        bgRemoverDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            bgRemoverDropZone.classList.remove('drag-active');
            if (e.dataTransfer.files.length > 0) {
                setBgRemoverFiles(e.dataTransfer.files);
            }
        });
    }

    if (btnRemoveBgRemoverFile) {
        btnRemoveBgRemoverFile.addEventListener('click', (e) => {
            e.stopPropagation();
            bgRemoverSelectedFiles = [];
            bgRemoverFileInput.value = '';
            updateBgRemoverUI();
        });
    }

    if (btnClearBgRemoverFiles) {
        btnClearBgRemoverFiles.addEventListener('click', () => {
            bgRemoverSelectedFiles = [];
            bgRemoverFileInput.value = '';
            updateBgRemoverUI();
        });
    }

    if (btnAddMoreBgRemoverFiles) {
        btnAddMoreBgRemoverFiles.addEventListener('click', () => {
            const tempInput = document.createElement('input');
            tempInput.type = 'file';
            tempInput.multiple = true;
            tempInput.accept = 'image/png, image/jpeg, image/webp, image/bmp';
            tempInput.onchange = () => {
                if (tempInput.files.length > 0) {
                    bgRemoverSelectedFiles = bgRemoverSelectedFiles.concat(Array.from(tempInput.files));
                    updateBgRemoverUI();
                }
            };
            tempInput.click();
        });
    }

    // Reuse from previous tab
    if (btnReuseForBgRemover) {
        btnReuseForBgRemover.addEventListener('click', () => {
            if (window.currentActiveResultUrl) {
                fetch(window.currentActiveResultUrl)
                    .then(res => res.blob())
                    .then(blob => {
                        const file = new File([blob], "last_processed_image.png", { type: blob.type || "image/png" });
                        setBgRemoverFiles([file]);
                        window.showToast("🔄 Đã lấy ảnh vừa xử lý vào tab Xóa Nền!");
                    })
                    .catch(() => window.showToast("⚠️ Chưa có ảnh nào vừa xử lý!"));
            } else {
                window.showToast("⚠️ Hãy xử lý một ảnh trước để sử dụng tính năng này!");
            }
        });
    }

    // Sample buttons
    document.querySelectorAll('.bg-sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleType = btn.dataset.sample;
            let sampleUrl = '/static/samples/sample_portrait_lowres.jpg';
            let name = 'sample_portrait.jpg';
            if (sampleType === 'dog') {
                sampleUrl = '/uploads/input_2de8d37ea0.png';
                name = 'sample_dog_sticker.png';
            }
            fetch(sampleUrl)
                .then(r => r.blob())
                .then(b => {
                    const file = new File([b], name, { type: b.type || 'image/jpeg' });
                    setBgRemoverFiles([file]);
                    window.showToast(`📸 Đã nạp ảnh ${btn.textContent.trim()}!`);
                });
        });
    });

    // Run Background Removal
    if (btnRunBgRemove) {
        btnRunBgRemove.addEventListener('click', async () => {
            if (bgRemoverSelectedFiles.length === 0) return;

            btnRunBgRemove.disabled = true;
            bgRemoverEmptyState.classList.add('hidden');
            bgRemoverCompareContainer.classList.add('hidden');
            bgResultOnlyContainer.classList.add('hidden');
            bgRemoverResultFooter.classList.add('hidden');
            bgBatchResultsContainer.classList.add('hidden');
            bgRemoverViewToggles.classList.add('hidden');

            bgRemoverProcessingState.classList.remove('hidden');
            bgRemoverStatusBadge.textContent = 'Đang xử lý AI...';
            bgRemoverStatusBadge.className = 'preview-badge badge-processing';

            const modelVal = bgModelSelect ? bgModelSelect.value : 'rmbg-1.4';
            const cleanModeVal = bgCleanModeSelect ? bgCleanModeSelect.value : 'deep_clean';
            const sensitivityVal = bgSensitivitySlider ? (parseFloat(bgSensitivitySlider.value) / 100.0) : 0.5;
            const deFringeVal = bgDeFringe && bgDeFringe.checked ? 'true' : 'false';

            const formData = new FormData();
            bgRemoverSelectedFiles.forEach(file => {
                formData.append('images', file);
            });
            formData.append('model_name', modelVal);
            formData.append('clean_mode', cleanModeVal);
            formData.append('sensitivity', sensitivityVal);
            formData.append('de_fringe', deFringeVal);
            formData.append('bg_color', selectedRemoverBgColor);
            formData.append('feather', bgFeatherEdge && bgFeatherEdge.checked ? 'true' : 'false');

            try {
                const response = await fetch('/api/remove-background', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                bgRemoverProcessingState.classList.add('hidden');

                if (result.status === 'success') {
                    bgRemoverStatusBadge.textContent = 'Hoàn tất';
                    bgRemoverStatusBadge.className = 'preview-badge badge-success';
                    window.showToast('✨ Tách nền hoàn tất!');

                    if (result.batch) {
                        // Batch view
                        bgBatchResultsContainer.classList.remove('hidden');
                        bgBatchSuccessCount.textContent = result.count;
                        if (btnDownloadBgZip) {
                            const zipUrl = result.download_url || `/api/download/${result.zip_filename}`;
                            btnDownloadBgZip.href = zipUrl;
                            btnDownloadBgZip.setAttribute('download', result.zip_filename || 'ClarityAI_Batch_NoBG.zip');
                            btnDownloadBgZip.onclick = async () => {
                                try {
                                    const fd = new FormData();
                                    fd.append('filename', result.zip_filename);
                                    await fetch('/api/save-to-downloads', { method: 'POST', body: fd });
                                } catch (_) {}
                                window.showToast(`📦 Đang tải file ZIP chung và lưu vào Downloads: ${result.zip_filename}`);
                            };
                        }

                        // Setup "Lưu Tất Cả Vào Downloads" button
                        const btnSaveBatchBgToDownloads = document.getElementById('btnSaveBatchBgToDownloads');
                        if (btnSaveBatchBgToDownloads) {
                            btnSaveBatchBgToDownloads.onclick = async () => {
                                btnSaveBatchBgToDownloads.disabled = true;
                                const oldHtml = btnSaveBatchBgToDownloads.innerHTML;
                                btnSaveBatchBgToDownloads.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang lưu...';

                                try {
                                    const res = await fetch('/api/save-batch-bg-to-downloads', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            items: result.results.map(r => ({
                                                filename: r.filename,
                                                original_filename: r.original_filename
                                            }))
                                        })
                                    });
                                    const json = await res.json();
                                    if (json.status === 'success') {
                                        window.showToast(`💾 ${json.message || 'Đã lưu tất cả ảnh vào Downloads!'}`);

                                        // Also trigger browser downloads for each image file
                                        result.results.forEach((item, idx) => {
                                            setTimeout(() => {
                                                const base = item.original_filename ? (item.original_filename.substring(0, item.original_filename.lastIndexOf('.')) || item.original_filename) : `image_${idx + 1}`;
                                                const name = `ClarityAI_${base}_nobg.png`;
                                                const a = document.createElement('a');
                                                a.href = `/api/download/${item.filename}?name=${encodeURIComponent(name)}`;
                                                a.download = name;
                                                document.body.appendChild(a);
                                                a.click();
                                                document.body.removeChild(a);
                                            }, idx * 250);
                                        });
                                    } else {
                                        window.showToast(`⚠️ Lỗi: ${json.message || 'Không thể lưu vào Downloads'}`);
                                    }
                                } catch (err) {
                                    window.showToast(`❌ Lỗi mạng: ${err.message}`);
                                } finally {
                                    btnSaveBatchBgToDownloads.disabled = false;
                                    btnSaveBatchBgToDownloads.innerHTML = oldHtml;
                                }
                            };
                        }

                        bgBatchResultsGrid.innerHTML = '';
                        result.results.forEach(item => {
                            const baseName = item.original_filename ? (item.original_filename.substring(0, item.original_filename.lastIndexOf('.')) || item.original_filename) : 'image';
                            const cleanPngName = `ClarityAI_${baseName}_nobg.png`;
                            const cardDownloadUrl = `/api/download/${item.filename}?name=${encodeURIComponent(cleanPngName)}`;

                            const card = document.createElement('div');
                            card.className = 'batch-result-card';
                            card.innerHTML = `
                                <div class="batch-result-img-box checkerboard-bg">
                                    <img src="${item.result_url}" alt="result" class="batch-result-img">
                                </div>
                                <div class="batch-result-info">
                                    <div class="batch-result-filename" title="${item.original_filename}">${item.original_filename}</div>
                                    <div class="batch-result-dims"><i class="fa-solid fa-check"></i> ${item.stats.duration_sec}s · PNG Trong Suốt</div>
                                    <div class="batch-result-bottom" style="width: 100%; margin-top: 6px;">
                                        <a href="${cardDownloadUrl}" download="${cleanPngName}" class="btn-card-download-png" title="Tải ảnh PNG và lưu trực tiếp vào Downloads">
                                            <i class="fa-solid fa-download"></i> Tải PNG (Downloads)
                                        </a>
                                    </div>
                                </div>
                            `;

                            // Add click listener for individual save & download feedback
                            const dlLink = card.querySelector('.btn-card-download-png');
                            if (dlLink) {
                                dlLink.onclick = async () => {
                                    try {
                                        const fd = new FormData();
                                        fd.append('filename', item.filename);
                                        fd.append('custom_name', cleanPngName);
                                        await fetch('/api/save-to-downloads', { method: 'POST', body: fd });
                                    } catch (_) {}
                                    window.showToast(`💾 Đã lưu và tải ảnh PNG: ${cleanPngName}`);
                                };
                            }

                            bgBatchResultsGrid.appendChild(card);
                        });
                    } else {
                        // Single file view
                        currentBgResultData = result.data;
                        window.currentActiveResultUrl = result.data.result_url;

                        bgRemoverViewToggles.classList.remove('hidden');
                        bgRemoverCompareContainer.classList.remove('hidden');
                        bgRemoverResultFooter.classList.remove('hidden');

                        bgOrigImg.src = result.data.original_url;
                        bgResultImg.src = result.data.result_url;
                        bgSingleResultImg.src = result.data.result_url;

                        // Reset slider
                        bgBeforeWrapper.style.width = '50%';
                        bgCompareHandle.style.left = '50%';

                        // Metadata
                        bgMetaDims.textContent = `${result.data.stats.final_width} x ${result.data.stats.final_height}`;
                        bgMetaDuration.textContent = `${result.data.stats.duration_sec}s`;
                        bgMetaModel.textContent = result.data.stats.model_name || 'RMBG-1.4 AI';
                        bgMetaSize.textContent = `${result.data.file_sizes.result_kb} KB`;

                        const singleBase = result.data.original_filename ? (result.data.original_filename.substring(0, result.data.original_filename.lastIndexOf('.')) || result.data.original_filename) : 'image';
                        const singlePngName = `ClarityAI_${singleBase}_nobg.png`;
                        const singleJpgName = `ClarityAI_${singleBase}_nobg.jpg`;

                        btnDownloadBgPng.href = `/api/download/${result.data.filename}?name=${encodeURIComponent(singlePngName)}`;
                        btnDownloadBgPng.setAttribute('download', singlePngName);
                        btnDownloadBgPng.onclick = async () => {
                            try {
                                const fd = new FormData();
                                fd.append('filename', result.data.filename);
                                fd.append('custom_name', singlePngName);
                                await fetch('/api/save-to-downloads', { method: 'POST', body: fd });
                            } catch (_) {}
                            window.showToast(`💾 Đã lưu và tải ảnh PNG: ${singlePngName}`);
                        };

                        btnDownloadBgJpg.href = result.data.download_jpg_url || `/api/download-jpg/${result.data.filename}`;
                        btnDownloadBgJpg.setAttribute('download', singleJpgName);

                        // Save to downloads
                        if (btnSaveBgToDownloads) {
                            btnSaveBgToDownloads.onclick = () => {
                                const fd = new FormData();
                                fd.append('filename', result.data.filename);
                                fd.append('custom_name', singlePngName);
                                fetch('/api/save-to-downloads', { method: 'POST', body: fd })
                                    .then(r => r.json())
                                    .then(res => {
                                        if (res.status === 'success') {
                                            window.showToast(`💾 Đã lưu vào Downloads: ${res.saved_name || res.filename}`);
                                        } else {
                                            window.showToast(`⚠️ Lỗi: ${res.message}`);
                                        }
                                    });
                            };
                        }
                    }
                } else {
                    bgRemoverStatusBadge.textContent = 'Lỗi xử lý';
                    bgRemoverStatusBadge.className = 'preview-badge badge-error';
                    bgRemoverEmptyState.classList.remove('hidden');
                    window.showToast(`❌ Lỗi: ${result.message || 'Không thể tách nền'}`);
                }
            } catch (err) {
                bgRemoverProcessingState.classList.add('hidden');
                bgRemoverEmptyState.classList.remove('hidden');
                bgRemoverStatusBadge.textContent = 'Lỗi kết nối';
                bgRemoverStatusBadge.className = 'preview-badge badge-error';
                window.showToast(`❌ Lỗi mạng: ${err.message}`);
            } finally {
                btnRunBgRemove.disabled = false;
            }
        });
    }

    // Comparison slider logic for Background Remover
    if (bgRemoverCompareContainer && bgBeforeWrapper && bgCompareHandle) {
        let isDraggingBgSlider = false;

        const updateBgSliderPos = (clientX) => {
            const rect = bgRemoverCompareContainer.getBoundingClientRect();
            let pos = ((clientX - rect.left) / rect.width) * 100;
            pos = Math.max(0, Math.min(100, pos));
            bgBeforeWrapper.style.width = `${pos}%`;
            bgCompareHandle.style.left = `${pos}%`;
        };

        bgCompareHandle.addEventListener('mousedown', () => { isDraggingBgSlider = true; });
        window.addEventListener('mouseup', () => { isDraggingBgSlider = false; });
        window.addEventListener('mousemove', (e) => {
            if (isDraggingBgSlider) updateBgSliderPos(e.clientX);
        });

        // Touch support
        bgCompareHandle.addEventListener('touchstart', () => { isDraggingBgSlider = true; });
        window.addEventListener('touchend', () => { isDraggingBgSlider = false; });
        window.addEventListener('touchmove', (e) => {
            if (isDraggingBgSlider && e.touches[0]) updateBgSliderPos(e.touches[0].clientX);
        });

        // Click on container to jump slider
        bgRemoverCompareContainer.addEventListener('click', (e) => {
            updateBgSliderPos(e.clientX);
        });
    }

    // View toggles (Compare vs Result Only vs Original)
    if (bgToggleCompare && bgToggleResultOnly && bgToggleOriginal) {
        bgToggleCompare.addEventListener('click', () => {
            bgToggleCompare.classList.add('active');
            bgToggleResultOnly.classList.remove('active');
            bgToggleOriginal.classList.remove('active');
            bgRemoverCompareContainer.classList.remove('hidden');
            bgResultOnlyContainer.classList.add('hidden');
            bgBeforeWrapper.style.width = '50%';
            bgCompareHandle.style.left = '50%';
        });

        bgToggleResultOnly.addEventListener('click', () => {
            bgToggleResultOnly.classList.add('active');
            bgToggleCompare.classList.remove('active');
            bgToggleOriginal.classList.remove('active');
            bgRemoverCompareContainer.classList.add('hidden');
            bgResultOnlyContainer.classList.remove('hidden');
        });

        bgToggleOriginal.addEventListener('click', () => {
            bgToggleOriginal.classList.add('active');
            bgToggleCompare.classList.remove('active');
            bgToggleResultOnly.classList.remove('active');
            bgRemoverCompareContainer.classList.remove('hidden');
            bgResultOnlyContainer.classList.add('hidden');
            bgBeforeWrapper.style.width = '100%';
            bgCompareHandle.style.left = '100%';
        });
    }

    // ==========================================
    // 16. IMAGE FORMAT CONVERTER CLIENT LOGIC
    // ==========================================
    const converterDropZone = document.getElementById('converterDropZone');
    const converterFileInput = document.getElementById('converterFileInput');
    const converterDropPrompt = document.getElementById('converterDropPrompt');
    const converterFileCountBadge = document.getElementById('converterFileCountBadge');

    const converterSinglePreviewWrapper = document.getElementById('converterSinglePreviewWrapper');
    const converterSingleThumb = document.getElementById('converterSingleThumb');
    const converterSingleFileName = document.getElementById('converterSingleFileName');
    const converterSingleFileMeta = document.getElementById('converterSingleFileMeta');
    const btnRemoveConverterSingle = document.getElementById('btnRemoveConverterSingle');

    const converterBatchListWrapper = document.getElementById('converterBatchListWrapper');
    const converterBatchSelectedCount = document.getElementById('converterBatchSelectedCount');
    const btnClearConverterFiles = document.getElementById('btnClearConverterFiles');
    const converterBatchFilesGrid = document.getElementById('converterBatchFilesGrid');
    const btnAddMoreConverterFiles = document.getElementById('btnAddMoreConverterFiles');

    const btnSampleConverterPng = document.getElementById('btnSampleConverterPng');
    const btnSampleConverterJpg = document.getElementById('btnSampleConverterJpg');
    const btnSampleConverterWebp = document.getElementById('btnSampleConverterWebp');

    const converterFormatCardsGrid = document.getElementById('converterFormatCardsGrid');
    const btnQualityLossless = document.getElementById('btnQualityLossless');
    const btnQualityHigh = document.getElementById('btnQualityHigh');
    const converterBgColorGroup = document.getElementById('converterBgColorGroup');
    const btnConverterBgWhite = document.getElementById('btnConverterBgWhite');
    const btnConverterBgBlack = document.getElementById('btnConverterBgBlack');
    const converterCustomColor = document.getElementById('converterCustomColor');
    const btnRunConvert = document.getElementById('btnRunConvert');

    const converterStatusBadge = document.getElementById('converterStatusBadge');
    const converterEmptyState = document.getElementById('converterEmptyState');
    const converterProcessingState = document.getElementById('converterProcessingState');
    const converterProcessingTitle = document.getElementById('converterProcessingTitle');
    const converterProcessingSub = document.getElementById('converterProcessingSub');

    const converterSingleResult = document.getElementById('converterSingleResult');
    const converterResultImg = document.getElementById('converterResultImg');
    const converterMetaName = document.getElementById('converterMetaName');
    const converterMetaDims = document.getElementById('converterMetaDims');
    const converterMetaSize = document.getElementById('converterMetaSize');
    const converterMetaTime = document.getElementById('converterMetaTime');
    const btnDownloadConvertedSingle = document.getElementById('btnDownloadConvertedSingle');
    const btnSaveConvertedSingleDownloads = document.getElementById('btnSaveConvertedSingleDownloads');

    const converterBatchResultsContainer = document.getElementById('converterBatchResultsContainer');
    const converterBatchSuccessCount = document.getElementById('converterBatchSuccessCount');
    const btnDownloadConverterZip = document.getElementById('btnDownloadConverterZip');
    const btnSaveBatchConverterToDownloads = document.getElementById('btnSaveBatchConverterToDownloads');
    const converterBatchResultsGrid = document.getElementById('converterBatchResultsGrid');

    let converterFiles = [];
    let selectedTargetFormat = 'png';
    let selectedQualityMode = 'lossless';
    let converterSelectedBgColor = 'white';

    // Format selection click handler
    if (converterFormatCardsGrid) {
        const cards = converterFormatCardsGrid.querySelectorAll('.format-card');
        cards.forEach(card => {
            card.addEventListener('click', () => {
                cards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                const radio = card.querySelector('input[type="radio"]');
                if (radio) radio.checked = true;
                selectedTargetFormat = card.dataset.fmt || 'png';

                // Show background color picker if target format has no alpha support
                if (['jpg', 'bmp', 'pdf'].includes(selectedTargetFormat)) {
                    if (converterBgColorGroup) converterBgColorGroup.classList.remove('hidden');
                } else {
                    if (converterBgColorGroup) converterBgColorGroup.classList.add('hidden');
                }
            });
        });
    }

    // Quality buttons
    if (btnQualityLossless && btnQualityHigh) {
        btnQualityLossless.addEventListener('click', () => {
            btnQualityLossless.classList.add('active');
            btnQualityHigh.classList.remove('active');
            selectedQualityMode = 'lossless';
        });
        btnQualityHigh.addEventListener('click', () => {
            btnQualityHigh.classList.add('active');
            btnQualityLossless.classList.remove('active');
            selectedQualityMode = 'high';
        });
    }

    // Background color presets for non-alpha formats
    if (btnConverterBgWhite && btnConverterBgBlack && converterCustomColor) {
        btnConverterBgWhite.addEventListener('click', () => {
            btnConverterBgWhite.classList.add('active');
            btnConverterBgBlack.classList.remove('active');
            converterSelectedBgColor = 'white';
        });
        btnConverterBgBlack.addEventListener('click', () => {
            btnConverterBgBlack.classList.add('active');
            btnConverterBgWhite.classList.remove('active');
            converterSelectedBgColor = 'black';
        });
        converterCustomColor.addEventListener('input', (e) => {
            btnConverterBgWhite.classList.remove('active');
            btnConverterBgBlack.classList.remove('active');
            converterSelectedBgColor = e.target.value;
        });
    }

    // Helper: update file upload views
    function updateConverterFilesUI() {
        if (!converterFiles || converterFiles.length === 0) {
            if (converterDropPrompt) converterDropPrompt.classList.remove('hidden');
            if (converterSinglePreviewWrapper) converterSinglePreviewWrapper.classList.add('hidden');
            if (converterBatchListWrapper) converterBatchListWrapper.classList.add('hidden');
            if (converterFileCountBadge) converterFileCountBadge.classList.add('hidden');
            return;
        }

        if (converterFiles.length === 1) {
            // Single file
            if (converterDropPrompt) converterDropPrompt.classList.add('hidden');
            if (converterBatchListWrapper) converterBatchListWrapper.classList.add('hidden');
            if (converterSinglePreviewWrapper) converterSinglePreviewWrapper.classList.remove('hidden');
            if (converterFileCountBadge) {
                converterFileCountBadge.classList.remove('hidden');
                converterFileCountBadge.textContent = '1 ảnh';
            }

            const f = converterFiles[0];
            if (converterSingleFileName) converterSingleFileName.textContent = f.name;
            const sizeKb = Math.round(f.size / 1024);
            const sizeStr = sizeKb > 1024 ? `${(sizeKb / 1024).toFixed(2)} MB` : `${sizeKb} KB`;

            const r = new FileReader();
            r.onload = (e) => {
                if (converterSingleThumb) converterSingleThumb.src = e.target.result;
                const tempImg = new Image();
                tempImg.onload = () => {
                    if (converterSingleFileMeta) converterSingleFileMeta.textContent = `${tempImg.naturalWidth} x ${tempImg.naturalHeight} · ${sizeStr}`;
                };
                tempImg.src = e.target.result;
            };
            r.readAsDataURL(f);
        } else {
            // Multi-file batch
            if (converterDropPrompt) converterDropPrompt.classList.add('hidden');
            if (converterSinglePreviewWrapper) converterSinglePreviewWrapper.classList.add('hidden');
            if (converterBatchListWrapper) converterBatchListWrapper.classList.remove('hidden');
            if (converterFileCountBadge) {
                converterFileCountBadge.classList.remove('hidden');
                converterFileCountBadge.textContent = `${converterFiles.length} ảnh`;
            }
            if (converterBatchSelectedCount) converterBatchSelectedCount.textContent = converterFiles.length;

            renderConverterBatchGrid();
        }
    }

    function renderConverterBatchGrid() {
        if (!converterBatchFilesGrid) return;
        converterBatchFilesGrid.innerHTML = '';

        converterFiles.forEach((file, idx) => {
            const card = document.createElement('div');
            card.className = 'batch-file-card';
            const sizeKb = Math.round(file.size / 1024);
            const sizeStr = sizeKb > 1024 ? `${(sizeKb / 1024).toFixed(1)} MB` : `${sizeKb} KB`;

            card.innerHTML = `
                <div class="batch-file-thumb-box">
                    <img src="" alt="${file.name}" class="batch-file-thumb" id="convThumb_${idx}">
                    <button type="button" class="btn-remove-batch-item" title="Xóa ảnh này" data-idx="${idx}">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>
                <div class="batch-file-info">
                    <div class="batch-file-name" title="${file.name}">${file.name}</div>
                    <div class="batch-file-meta">#${idx + 1} · ${sizeStr}</div>
                </div>
            `;

            const imgEl = card.querySelector(`#convThumb_${idx}`);
            const fr = new FileReader();
            fr.onload = (ev) => { if (imgEl) imgEl.src = ev.target.result; };
            fr.readAsDataURL(file);

            const btnRemove = card.querySelector('.btn-remove-batch-item');
            btnRemove.addEventListener('click', (e) => {
                e.stopPropagation();
                converterFiles.splice(idx, 1);
                updateConverterFilesUI();
            });

            converterBatchFilesGrid.appendChild(card);
        });
    }

    function handleConverterFiles(newFiles) {
        if (!newFiles || newFiles.length === 0) return;
        const valid = Array.from(newFiles).filter(f => f.type.startsWith('image/') || f.name.match(/\.(png|jpe?g|webp|bmp|tiff?|ico|pdf|avif|gif)$/i));
        if (valid.length === 0) {
            window.showToast('⚠️ Vui lòng chọn tệp ảnh hợp lệ');
            return;
        }

        if (converterFiles.length === 0) {
            converterFiles = valid;
        } else {
            converterFiles = converterFiles.concat(valid);
        }
        updateConverterFilesUI();
    }

    // Dropzone events
    if (converterDropZone && converterFileInput) {
        converterDropZone.addEventListener('click', (e) => {
            if (e.target.closest('#btnRemoveConverterSingle')) return;
            converterFileInput.click();
        });

        converterFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleConverterFiles(e.target.files);
            }
        });

        ['dragenter', 'dragover'].forEach(name => {
            converterDropZone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                converterDropZone.classList.add('drag-active');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            converterDropZone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                converterDropZone.classList.remove('drag-active');
            });
        });

        converterDropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleConverterFiles(e.dataTransfer.files);
            }
        });
    }

    if (btnRemoveConverterSingle) {
        btnRemoveConverterSingle.addEventListener('click', (e) => {
            e.stopPropagation();
            converterFiles = [];
            if (converterFileInput) converterFileInput.value = '';
            updateConverterFilesUI();
        });
    }

    if (btnClearConverterFiles) {
        btnClearConverterFiles.addEventListener('click', () => {
            converterFiles = [];
            if (converterFileInput) converterFileInput.value = '';
            updateConverterFilesUI();
        });
    }

    if (btnAddMoreConverterFiles && converterFileInput) {
        btnAddMoreConverterFiles.addEventListener('click', () => {
            converterFileInput.click();
        });
    }

    // Sample loading helper
    async function loadConverterSample(url, defaultName) {
        try {
            window.showToast('📥 Đang tải ảnh mẫu...');
            const r = await fetch(url);
            const blob = await r.blob();
            const f = new File([blob], defaultName, { type: blob.type });
            converterFiles = [f];
            updateConverterFilesUI();
            window.showToast(`✨ Đã nạp ảnh mẫu: ${defaultName}`);
        } catch (err) {
            window.showToast(`❌ Lỗi tải ảnh mẫu: ${err.message}`);
        }
    }

    if (btnSampleConverterPng) {
        btnSampleConverterPng.addEventListener('click', () => {
            loadConverterSample('/static/samples/sample_anime_lowres.jpg', 'sample_anime_art.jpg');
        });
    }
    if (btnSampleConverterJpg) {
        btnSampleConverterJpg.addEventListener('click', () => {
            loadConverterSample('/static/samples/sample_portrait_lowres.jpg', 'sample_portrait_hd.jpg');
        });
    }
    if (btnSampleConverterWebp) {
        btnSampleConverterWebp.addEventListener('click', () => {
            loadConverterSample('/static/samples/sample_anime_lowres.jpg', 'sample_sticker.png');
        });
    }

    // Run Conversion Action
    if (btnRunConvert) {
        btnRunConvert.addEventListener('click', async () => {
            if (!converterFiles || converterFiles.length === 0) {
                window.showToast('⚠️ Vui lòng chọn hoặc kéo thả ít nhất 1 ảnh để chuyển đổi!');
                return;
            }

            btnRunConvert.disabled = true;
            const originalBtnHtml = btnRunConvert.innerHTML;
            btnRunConvert.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang Chuyển Đổi Định Dạng...';

            // Show processing UI
            if (converterEmptyState) converterEmptyState.classList.add('hidden');
            if (converterSingleResult) converterSingleResult.classList.add('hidden');
            if (converterBatchResultsContainer) converterBatchResultsContainer.classList.add('hidden');
            if (converterProcessingState) converterProcessingState.classList.remove('hidden');
            if (converterStatusBadge) {
                converterStatusBadge.textContent = 'Đang xử lý';
                converterStatusBadge.className = 'preview-badge badge-processing';
            }

            const fd = new FormData();
            fd.append('target_format', selectedTargetFormat);
            fd.append('quality_mode', selectedQualityMode);
            fd.append('bg_color', converterSelectedBgColor);

            converterFiles.forEach(f => {
                fd.append('images', f);
            });

            try {
                const res = await fetch('/api/convert-format', {
                    method: 'POST',
                    body: fd
                });
                const result = await res.json();

                if (converterProcessingState) converterProcessingState.classList.add('hidden');

                if (result.status === 'success') {
                    if (converterStatusBadge) {
                        converterStatusBadge.textContent = 'Hoàn tất';
                        converterStatusBadge.className = 'preview-badge badge-success';
                    }
                    window.showToast(`🎉 Chuyển đổi thành công sang ${result.target_format || selectedTargetFormat.toUpperCase()}!`);

                    if (result.batch) {
                        // Multi-image batch result
                        if (converterBatchResultsContainer) converterBatchResultsContainer.classList.remove('hidden');
                        if (converterBatchSuccessCount) converterBatchSuccessCount.textContent = result.count;

                        // Setup ZIP download button
                        if (btnDownloadConverterZip) {
                            const zipUrl = result.download_url || `/api/download/${result.zip_filename}`;
                            btnDownloadConverterZip.href = zipUrl;
                            btnDownloadConverterZip.setAttribute('download', result.zip_filename || `Converted_${selectedTargetFormat.toUpperCase()}.zip`);
                            btnDownloadConverterZip.onclick = async () => {
                                try {
                                    const sfd = new FormData();
                                    sfd.append('filename', result.zip_filename);
                                    await fetch('/api/save-to-downloads', { method: 'POST', body: sfd });
                                } catch (_) {}
                                window.showToast(`📦 Đang tải file ZIP chung và lưu vào Downloads: ${result.zip_filename}`);
                            };
                        }

                        // Setup Save All to Downloads button
                        if (btnSaveBatchConverterToDownloads) {
                            btnSaveBatchConverterToDownloads.onclick = async () => {
                                btnSaveBatchConverterToDownloads.disabled = true;
                                const oldHtml = btnSaveBatchConverterToDownloads.innerHTML;
                                btnSaveBatchConverterToDownloads.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang lưu...';

                                try {
                                    const saveRes = await fetch('/api/save-batch-convert-to-downloads', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            items: result.results.map(r => ({
                                                filename: r.filename,
                                                clean_filename: r.clean_filename,
                                                original_filename: r.original_filename
                                            }))
                                        })
                                    });
                                    const saveJson = await saveRes.json();
                                    if (saveJson.status === 'success') {
                                        window.showToast(`💾 ${saveJson.message || 'Đã lưu tất cả ảnh vào Downloads!'}`);

                                        // Trigger browser downloads with stagger
                                        result.results.forEach((item, idx) => {
                                            setTimeout(() => {
                                                const a = document.createElement('a');
                                                a.href = `/api/download/${item.filename}?name=${encodeURIComponent(item.clean_filename)}`;
                                                a.download = item.clean_filename;
                                                document.body.appendChild(a);
                                                a.click();
                                                document.body.removeChild(a);
                                            }, idx * 250);
                                        });
                                    } else {
                                        window.showToast(`⚠️ Lỗi: ${saveJson.message || 'Không thể lưu vào Downloads'}`);
                                    }
                                } catch (err) {
                                    window.showToast(`❌ Lỗi mạng: ${err.message}`);
                                } finally {
                                    btnSaveBatchConverterToDownloads.disabled = false;
                                    btnSaveBatchConverterToDownloads.innerHTML = oldHtml;
                                }
                            };
                        }

                        // Populate batch grid
                        if (converterBatchResultsGrid) {
                            converterBatchResultsGrid.innerHTML = '';
                            result.results.forEach(item => {
                                const card = document.createElement('div');
                                card.className = 'batch-result-card';
                                const cardDlUrl = `/api/download/${item.filename}?name=${encodeURIComponent(item.clean_filename)}`;

                                card.innerHTML = `
                                    <div class="batch-result-img-box checkerboard-bg">
                                        <img src="${item.result_url}" alt="result" class="batch-result-img">
                                    </div>
                                    <div class="batch-result-info">
                                        <div class="batch-result-filename" title="${item.clean_filename}">${item.clean_filename}</div>
                                        <div class="batch-result-dims">
                                            <i class="fa-solid fa-check"></i> ${item.stats.new_size_kb} KB (${item.stats.percent_diff > 0 ? '+' : ''}${item.stats.percent_diff}%) · ${item.stats.duration_sec}s
                                        </div>
                                        <div class="batch-result-bottom" style="width: 100%; margin-top: 6px;">
                                            <a href="${cardDlUrl}" download="${item.clean_filename}" class="btn-card-download-png" title="Tải ảnh và lưu vào Downloads">
                                                <i class="fa-solid fa-download"></i> Tải Về (Downloads)
                                            </a>
                                        </div>
                                    </div>
                                `;

                                const dlLink = card.querySelector('.btn-card-download-png');
                                if (dlLink) {
                                    dlLink.onclick = async () => {
                                        try {
                                            const sfd = new FormData();
                                            sfd.append('filename', item.filename);
                                            sfd.append('custom_name', item.clean_filename);
                                            await fetch('/api/save-to-downloads', { method: 'POST', body: sfd });
                                        } catch (_) {}
                                        window.showToast(`💾 Đã lưu và tải ảnh: ${item.clean_filename}`);
                                    };
                                }

                                converterBatchResultsGrid.appendChild(card);
                            });
                        }
                    } else {
                        // Single file result
                        if (converterSingleResult) converterSingleResult.classList.remove('hidden');

                        const d = result.data;
                        if (converterResultImg) converterResultImg.src = d.result_url;
                        if (converterMetaName) converterMetaName.textContent = d.clean_filename;
                        if (converterMetaDims) converterMetaDims.textContent = `${d.stats.new_width} x ${d.stats.new_height}`;
                        if (converterMetaSize) {
                            converterMetaSize.textContent = `${d.stats.orig_size_kb} KB → ${d.stats.new_size_kb} KB (${d.stats.percent_diff > 0 ? '+' : ''}${d.stats.percent_diff}%)`;
                        }
                        if (converterMetaTime) converterMetaTime.textContent = `${d.stats.duration_sec}s`;

                        // Single download link
                        if (btnDownloadConvertedSingle) {
                            const singleDlUrl = `/api/download/${d.filename}?name=${encodeURIComponent(d.clean_filename)}`;
                            btnDownloadConvertedSingle.href = singleDlUrl;
                            btnDownloadConvertedSingle.setAttribute('download', d.clean_filename);
                            btnDownloadConvertedSingle.innerHTML = `<i class="fa-solid fa-download"></i> Tải Ảnh (${d.clean_filename})`;
                            btnDownloadConvertedSingle.onclick = async () => {
                                try {
                                    const sfd = new FormData();
                                    sfd.append('filename', d.filename);
                                    sfd.append('custom_name', d.clean_filename);
                                    await fetch('/api/save-to-downloads', { method: 'POST', body: sfd });
                                } catch (_) {}
                                window.showToast(`💾 Đã lưu và tải ảnh: ${d.clean_filename}`);
                            };
                        }

                        // Single save to Downloads button
                        if (btnSaveConvertedSingleDownloads) {
                            btnSaveConvertedSingleDownloads.onclick = async () => {
                                try {
                                    const sfd = new FormData();
                                    sfd.append('filename', d.filename);
                                    sfd.append('custom_name', d.clean_filename);
                                    const r = await fetch('/api/save-to-downloads', { method: 'POST', body: sfd });
                                    const j = await r.json();
                                    if (j.status === 'success') {
                                        window.showToast(`💾 ${j.message || 'Đã lưu vào thư mục Downloads!'}`);
                                    } else {
                                        window.showToast(`⚠️ ${j.message}`);
                                    }
                                } catch (e) {
                                    window.showToast(`❌ Lỗi lưu tệp: ${e.message}`);
                                }
                            };
                        }
                    }
                } else {
                    if (converterStatusBadge) {
                        converterStatusBadge.textContent = 'Lỗi xử lý';
                        converterStatusBadge.className = 'preview-badge badge-error';
                    }
                    if (converterEmptyState) converterEmptyState.classList.remove('hidden');
                    window.showToast(`❌ Lỗi: ${result.message || 'Không thể chuyển đổi định dạng'}`);
                }
            } catch (err) {
                if (converterProcessingState) converterProcessingState.classList.add('hidden');
                if (converterEmptyState) converterEmptyState.classList.remove('hidden');
                if (converterStatusBadge) {
                    converterStatusBadge.textContent = 'Lỗi kết nối';
                    converterStatusBadge.className = 'preview-badge badge-error';
                }
                window.showToast(`❌ Lỗi mạng: ${err.message}`);
            } finally {
                btnRunConvert.disabled = false;
                btnRunConvert.innerHTML = originalBtnHtml;
            }
        });
    }
});




