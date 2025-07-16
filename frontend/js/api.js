/**
 * Upload Module
 * Handles file upload functionality and validation
 */

class UploadManager {
    constructor() {
        this.selectedFiles = [];
        this.isDragging = false;
        this.init();
    }

    init() {
        this.setupElements();
        this.attachEventListeners();
    }

    setupElements() {
        // Get DOM elements
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.selectedFilesDiv = document.getElementById('selectedFiles');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.uploadError = document.getElementById('uploadError');

        // Validate required elements
        if (!this.fileInput || !this.uploadArea || !this.uploadBtn) {
            Utils.logger.error('Required upload elements not found');
            return;
        }
    }

    attachEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });

        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Drag and drop events
        this.setupDragAndDrop();

        // Upload button click
        this.uploadBtn.addEventListener('click', () => {
            this.handleUpload();
        });
    }

    setupDragAndDrop() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('dragover');
                this.isDragging = true;
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
                this.isDragging = false;
            }, false);
        });

        // Handle dropped files
        this.uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            this.handleFileSelection(files);
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelection(files) {
        Utils.logger.info('Files selected:', files.length);
        
        // Validate files
        const validation = Utils.validateFiles(files);
        
        if (!validation.valid) {
            this.showError(validation.errors.join(', '));
            return;
        }

        // Store valid files
        this.selectedFiles = validation.validFiles;
        
        // Update UI
        this.displaySelectedFiles();
        this.updateUploadButton();
        this.hideError();

        Utils.logger.info('Valid files selected:', this.selectedFiles.length);
    }

    displaySelectedFiles() {
        if (this.selectedFiles.length === 0) {
            this.selectedFilesDiv.classList.add('hidden');
            return;
        }

        const filesHTML = this.selectedFiles.map((file, index) => `
            <div class="file-item" data-index="${index}">
                <svg class="file-icon" viewBox="0 0 24 24">
                    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                </svg>
                <div class="file-details">
                    <div class="file-name">${Utils.escapeHtml(file.name)}</div>
                    <div class="file-size">${Utils.formatFileSize(file.size)}</div>
                </div>
                <button class="remove-file-btn" onclick="uploadManager.removeFile(${index})" title="Remove file">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
                    </svg>
                </button>
            </div>
        `).join('');

        this.selectedFilesDiv.innerHTML = filesHTML;
        this.selectedFilesDiv.classList.remove('hidden');
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.displaySelectedFiles();
        this.updateUploadButton();
        
        if (this.selectedFiles.length === 0) {
            this.hideError();
        }
    }

    updateUploadButton() {
        const hasFiles = this.selectedFiles.length > 0;
        this.uploadBtn.disabled = !hasFiles;
        
        if (hasFiles) {
            const fileCount = this.selectedFiles.length;
            const totalSize = this.selectedFiles.reduce((sum, file) => sum + file.size, 0);
            this.uploadBtn.textContent = `Process ${fileCount} Document${fileCount > 1 ? 's' : ''} (${Utils.formatFileSize(totalSize)})`;
        } else {
            this.uploadBtn.textContent = 'Process Documents';
        }
    }

    async handleUpload() {
        if (this.selectedFiles.length === 0) {
            this.showError('Please select files to upload');
            return;
        }

        Utils.logger.info('Starting upload process...');
        
        try {
            // Disable UI
            this.setUploadState(true);
            this.hideError();

            // Upload files
            const result = await apiClient.uploadDocuments(this.selectedFiles);
            
            Utils.logger.info('Upload successful:', result);
            
            // Start processing
            if (result.process_id) {
                processingManager.startProcessing(result.process_id, this.selectedFiles);
            } else {
                throw new Error('No process ID received from server');
            }

        } catch (error) {
            Utils.logger.error('Upload failed:', error);
            this.showError(error.message || ERROR_MESSAGES.UPLOAD_FAILED);
            this.setUploadState(false);
        }
    }

    setUploadState(uploading) {
        this.uploadBtn.disabled = uploading;
        this.fileInput.disabled = uploading;
        this.uploadArea.style.pointerEvents = uploading ? 'none' : 'auto';
        
        if (uploading) {
            this.uploadBtn.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="spinner" style="width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    Uploading...
                </div>
            `;
            this.uploadArea.classList.add('loading');
        } else {
            this.updateUploadButton();
            this.uploadArea.classList.remove('loading');
        }
    }

    showError(message) {
        if (this.uploadError) {
            this.uploadError.textContent = message;
            this.uploadError.classList.remove('hidden');
        }
    }

    hideError() {
        if (this.uploadError) {
            this.uploadError.classList.add('hidden');
        }
    }

    reset() {
        this.selectedFiles = [];
        this.fileInput.value = '';
        this.selectedFilesDiv.classList.add('hidden');
        this.updateUploadButton();
        this.hideError();
        this.setUploadState(false);
    }

    // Public methods for external use
    getSelectedFiles() {
        return [...this.selectedFiles];
    }

    addFiles(files) {
        const validation = Utils.validateFiles([...this.selectedFiles, ...files]);
        
        if (validation.valid) {
            this.selectedFiles = validation.validFiles;
            this.displaySelectedFiles();
            this.updateUploadButton();
            this.hideError();
            return true;
        } else {
            this.showError(validation.errors.join(', '));
            return false;
        }
    }

    clearFiles() {
        this.reset();
    }

    validateCurrentFiles() {
        const validation = Utils.validateFiles(this.selectedFiles);
        
        if (!validation.valid) {
            this.showError(validation.errors.join(', '));
            this.selectedFiles = validation.validFiles;
            this.displaySelectedFiles();
            this.updateUploadButton();
        }
        
        return validation.valid;
    }
}

// Enhanced file validation with more detailed feedback
class FileValidator {
    static validate(file) {
        const errors = [];
        const warnings = [];

        // Basic validation
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            errors.push(`${file.name}: Invalid file type. Only PDF files are supported.`);
        }

        if (file.size > CONFIG.UPLOAD.MAX_FILE_SIZE) {
            errors.push(`${file.name}: File too large (${Utils.formatFileSize(file.size)}). Maximum size is ${Utils.formatFileSize(CONFIG.UPLOAD.MAX_FILE_SIZE)}.`);
        }

        if (file.size === 0) {
            errors.push(`${file.name}: File is empty.`);
        }

        // Size warnings
        if (file.size > 10 * 1024 * 1024) { // 10MB
            warnings.push(`${file.name}: Large file (${Utils.formatFileSize(file.size)}) may take longer to process.`);
        }

        // Name validation
        if (file.name.length > 255) {
            warnings.push(`${file.name}: Very long filename may cause issues.`);
        }

        return {
            valid: errors.length === 0,
            errors,
            warnings
        };
    }

    static validateBatch(files) {
        const fileArray = Array.from(files);
        const allErrors = [];
        const allWarnings = [];
        const validFiles = [];

        // Check file count
        if (fileArray.length > CONFIG.UPLOAD.MAX_FILES) {
            allErrors.push(`Too many files selected (${fileArray.length}). Maximum is ${CONFIG.UPLOAD.MAX_FILES}.`);
        }

        // Check total size
        const totalSize = fileArray.reduce((sum, file) => sum + file.size, 0);
        const maxTotalSize = CONFIG.UPLOAD.MAX_FILE_SIZE * CONFIG.UPLOAD.MAX_FILES;
        
        if (totalSize > maxTotalSize) {
            allErrors.push(`Total file size too large (${Utils.formatFileSize(totalSize)}). Maximum total size is ${Utils.formatFileSize(maxTotalSize)}.`);
        }

        // Validate individual files
        fileArray.forEach(file => {
            const validation = this.validate(file);
            allErrors.push(...validation.errors);
            allWarnings.push(...validation.warnings);
            
            if (validation.valid) {
                validFiles.push(file);
            }
        });

        return {
            valid: allErrors.length === 0,
            errors: allErrors,
            warnings: allWarnings,
            validFiles,
            totalSize,
            fileCount: fileArray.length
        };
    }
}

// Create global upload manager instance
const uploadManager = new UploadManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UploadManager, FileValidator, uploadManager };
}