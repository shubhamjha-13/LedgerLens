/**
 * Processing Manager
 * Handles document processing workflow and UI
 */

class ProcessingManager {
    constructor() {
        this.currentProcessId = null;
        this.selectedFiles = [];
        this.processingStatus = null;
        this.particleAnimation = null;
        this.statusCheckInterval = null;
        this.init();
    }

    init() {
        this.setupElements();
    }

    setupElements() {
        // Processing overlay and modal
        this.processingOverlay = document.getElementById('processingOverlay');
        this.processingModal = this.processingOverlay?.querySelector('.processing-modal');
        
        // Modal content elements
        this.processingModalMessage = document.getElementById('processingModalMessage');
        this.processingModalSubmessage = document.getElementById('processingModalSubmessage');
        this.documentList = document.getElementById('documentList');
        this.progressModalFill = document.getElementById('progressModalFill');
        this.progressModalText = document.getElementById('progressModalText');
        this.progressModalDetails = document.getElementById('progressModalDetails');
        this.processingModalError = document.getElementById('processingModalError');
        this.floatingParticles = document.getElementById('floatingParticles');

        // Validate required elements
        if (!this.processingOverlay) {
            Utils.logger.error('Processing overlay element not found');
        }
    }

    /**
     * Start processing workflow
     */
    async startProcessing(processId, files) {
        Utils.logger.info('Starting processing workflow:', processId);
        
        this.currentProcessId = processId;
        this.selectedFiles = files;
        
        // Show processing modal
        await this.showProcessingModal();
        
        // Initialize UI state
        this.initializeProcessingUI();
        
        // Start monitoring
        this.startMonitoring();
    }

    /**
     * Show processing modal with animation
     */
    async showProcessingModal() {
        if (!this.processingOverlay) return;

        await animationManager.showProcessingModal(this.processingOverlay);
        
        // Create particle animation
        if (this.floatingParticles) {
            this.particleAnimation = animationManager.createFloatingParticles(
                this.floatingParticles,
                {
                    count: 20,
                    lifetime: 8000,
                    interval: 800,
                    colors: ['#667eea', '#764ba2']
                }
            );
        }
    }

    /**
     * Hide processing modal with animation
     */
    async hideProcessingModal() {
        if (!this.processingOverlay) return;

        // Stop particle animation
        if (this.particleAnimation) {
            this.particleAnimation.stop();
            this.particleAnimation = null;
        }

        await animationManager.hideProcessingModal(this.processingOverlay);
        
        // Reset modal state
        this.resetModalState();
    }

    /**
     * Initialize processing UI
     */
    initializeProcessingUI() {
        // Set initial messages
        if (this.processingModalMessage) {
            this.processingModalMessage.textContent = 'Processing your financial documents...';
        }
        
        if (this.processingModalSubmessage) {
            const fileCount = this.selectedFiles.length;
            this.processingModalSubmessage.textContent = 
                `Analyzing ${fileCount} document${fileCount > 1 ? 's' : ''} with advanced AI`;
        }

        // Update document list
        this.updateDocumentList();

        // Reset progress
        this.updateProgress(0, 'Initializing...');

        // Hide error
        this.hideError();
    }

    /**
     * Update document list UI
     */
    updateDocumentList() {
        if (!this.documentList || !this.selectedFiles) return;

        const documentsHTML = this.selectedFiles.map((file, index) => `
            <div class="document-item" id="doc-${index}">
                <div class="document-status pending" id="status-${index}"></div>
                <div class="document-name">${Utils.escapeHtml(file.name)}</div>
            </div>
        `).join('');

        this.documentList.innerHTML = documentsHTML;
    }

    /**
     * Update document status
     */
    updateDocumentStatus(index, status, isProcessing = false) {
        const statusElement = document.getElementById(`status-${index}`);
        const itemElement = document.getElementById(`doc-${index}`);
        
        if (!statusElement || !itemElement) return;

        // Update status class
        statusElement.className = `document-status ${status}`;
        
        // Update processing state
        if (isProcessing) {
            itemElement.classList.add('processing');
        } else {
            itemElement.classList.remove('processing');
        }

        // Animate status change
        animationManager.animateDocumentStatus(itemElement, status);
    }

    /**
     * Update progress display
     */
    updateProgress(percentage, details = '') {
        // Update progress bar
        if (this.progressModalFill) {
            animationManager.animateProgress(this.progressModalFill, percentage, 800);
        }

        // Update progress text
        if (this.progressModalText) {
            const span = this.progressModalText.querySelector('span');
            if (span) {
                span.textContent = `${Math.round(percentage)}% complete`;
            }
        }

        // Update details
        if (this.progressModalDetails && details) {
            this.progressModalDetails.textContent = details;
        }
    }

    /**
     * Update processing message based on progress
     */
    updateProcessingMessage(progress) {
        if (!this.processingModalMessage) return;

        let message;
        if (progress < 20) {
            message = 'Analyzing document structure...';
        } else if (progress < 40) {
            message = 'Extracting financial entities...';
        } else if (progress < 60) {
            message = 'Building knowledge relationships...';
        } else if (progress < 80) {
            message = 'Creating vector embeddings...';
        } else if (progress < 95) {
            message = 'Optimizing retrieval system...';
        } else {
            message = 'Almost ready!';
        }

        this.processingModalMessage.textContent = message;
    }

    /**
     * Start monitoring processing status
     */
    startMonitoring() {
        if (!this.currentProcessId) return;

        Utils.logger.info('Starting processing monitoring');

        // Use API client monitoring with callbacks
        apiClient.monitorProcessing(
            this.currentProcessId,
            (status) => this.handleStatusUpdate(status),
            (status) => this.handleProcessingComplete(status),
            (error) => this.handleProcessingError(error)
        );
    }

    /**
     * Handle status update from API
     */
    handleStatusUpdate(status) {
        Utils.logger.debug('Processing status update:', status);
        
        this.processingStatus = status;
        
        // Update progress
        this.updateProgress(status.progress || 0, status.message || '');
        
        // Update processing message
        this.updateProcessingMessage(status.progress || 0);

        // Handle current document
        if (status.current_document) {
            this.handleCurrentDocument(status.current_document);
        }

        // Update details based on progress
        this.updateProgressDetails(status);
    }

    /**
     * Handle current document being processed
     */
    handleCurrentDocument(currentDocument) {
        // Update details
        if (this.progressModalDetails) {
            this.progressModalDetails.textContent = `Processing: ${currentDocument}`;
        }

        // Find and update document status
        const docIndex = this.selectedFiles.findIndex(file => 
            currentDocument.includes(file.name.replace('.pdf', ''))
        );
        
        if (docIndex >= 0) {
            // Mark previous documents as completed
            for (let i = 0; i < docIndex; i++) {
                this.updateDocumentStatus(i, 'completed');
            }
            
            // Mark current as processing
            this.updateDocumentStatus(docIndex, 'processing', true);
        }
    }

    /**
     * Update progress details based on status
     */
    updateProgressDetails(status) {
        if (!this.progressModalDetails) return;

        const progress = status.progress || 0;
        
        if (status.current_document) {
            // Currently processing a document
            return; // Already handled in handleCurrentDocument
        } else if (progress < 10) {
            this.progressModalDetails.textContent = 'Initializing system...';
        } else if (progress < 95) {
            this.progressModalDetails.textContent = 'Building knowledge graph...';
        } else {
            this.progressModalDetails.textContent = 'Finalizing setup...';
        }
    }

    /**
     * Handle processing completion
     */
    async handleProcessingComplete(status) {
        Utils.logger.info('Processing completed successfully:', status);

        // Mark all documents as completed
        this.selectedFiles.forEach((_, index) => {
            this.updateDocumentStatus(index, 'completed');
        });

        // Show success state
        this.updateProgress(100, 'System ready!');
        
        if (this.processingModalMessage) {
            this.processingModalMessage.textContent = 'Your Financial RAG is ready to use!';
        }
        
        if (this.processingModalSubmessage) {
            this.processingModalSubmessage.textContent = 'Transitioning to chat interface...';
        }

        // Add celebration effect
        if (this.processingModal) {
            animationManager.createCelebration(this.processingModal, {
                emoji: '🎉',
                duration: 800,
                size: '3rem'
            });
            
            // Add success animation to modal
            this.processingModal.classList.add('success-animation');
        }

        // Wait for celebration, then transition
        await Utils.wait(2500);
        
        // Enhanced transition to chat
        await this.transitionToChat(status);
    }

    /**
     * Enhanced transition to chat stage
     */
    async transitionToChat(status) {
        // Fade out modal with enhanced animation
        if (this.processingOverlay) {
            this.processingOverlay.style.transition = 'all 0.8s ease';
            this.processingOverlay.style.opacity = '0';
            this.processingOverlay.style.transform = 'scale(0.9)';
            
            await Utils.wait(400);
        }

        // Hide processing modal
        await this.hideProcessingModal();

        // Get stage elements
        const uploadStage = document.getElementById('uploadStage');
        const chatStage = document.getElementById('chatStage');

        if (uploadStage && chatStage) {
            // Smooth transition between stages
            uploadStage.classList.add('stage-transition-out');
            
            await Utils.wait(200);
            
            uploadStage.classList.add('hidden');
            chatStage.classList.remove('hidden');
            chatStage.classList.add('stage-transition-in');
            
            await Utils.wait(100);
            
            chatStage.classList.add('active');
            
            // Load system stats
            if (typeof systemManager !== 'undefined') {
                systemManager.loadSystemStats();
            }
            
            // Focus chat input
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.focus();
            }
            
            // Show enhanced welcome message
            await Utils.wait(300);
            
            if (typeof chatManager !== 'undefined') {
                chatManager.addWelcomeMessage(status);
            }
        }
    }

    /**
     * Handle processing error
     */
    handleProcessingError(error) {
        Utils.logger.error('Processing error:', error);

        // Mark failed documents
        this.selectedFiles.forEach((_, index) => {
            this.updateDocumentStatus(index, 'error');
        });

        // Show error message
        this.showError(error.message || ERROR_MESSAGES.PROCESSING_FAILED);

        // Stop particle animation
        if (this.particleAnimation) {
            this.particleAnimation.stop();
            this.particleAnimation = null;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        if (this.processingModalError) {
            this.processingModalError.textContent = message;
            this.processingModalError.classList.remove('hidden');
        }
    }

    /**
     * Hide error message
     */
    hideError() {
        if (this.processingModalError) {
            this.processingModalError.classList.add('hidden');
        }
    }

    /**
     * Reset modal state
     */
    resetModalState() {
        // Reset progress
        if (this.progressModalFill) {
            this.progressModalFill.style.width = '0%';
        }

        if (this.progressModalText) {
            const span = this.progressModalText.querySelector('span');
            if (span) {
                span.textContent = '0% complete';
            }
        }

        if (this.progressModalDetails) {
            this.progressModalDetails.textContent = 'Initializing...';
        }

        // Clear document list
        if (this.documentList) {
            this.documentList.innerHTML = '';
        }

        // Remove success animation
        if (this.processingModal) {
            this.processingModal.classList.remove('success-animation');
        }

        // Reset overlay styles
        if (this.processingOverlay) {
            this.processingOverlay.style.transition = '';
            this.processingOverlay.style.opacity = '';
            this.processingOverlay.style.transform = '';
        }

        // Hide error
        this.hideError();

        // Clear state
        this.currentProcessId = null;
        this.selectedFiles = [];
        this.processingStatus = null;
    }

    /**
     * Cancel processing
     */
    async cancelProcessing() {
        Utils.logger.info('Cancelling processing');

        // Stop monitoring
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }

        // Stop animations
        if (this.particleAnimation) {
            this.particleAnimation.stop();
            this.particleAnimation = null;
        }

        // Hide modal
        await this.hideProcessingModal();

        // Reset upload manager
        if (typeof uploadManager !== 'undefined') {
            uploadManager.reset();
        }
    }

    /**
     * Get current processing status
     */
    getCurrentStatus() {
        return {
            processId: this.currentProcessId,
            files: this.selectedFiles,
            status: this.processingStatus
        };
    }

    /**
     * Check if currently processing
     */
    isProcessing() {
        return this.currentProcessId !== null;
    }

    /**
     * Retry processing
     */
    async retryProcessing() {
        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            Utils.logger.error('No files to retry processing');
            return;
        }

        Utils.logger.info('Retrying processing');

        try {
            // Reset state
            this.resetModalState();

            // Re-upload files
            const result = await apiClient.uploadDocuments(this.selectedFiles);
            
            if (result.process_id) {
                await this.startProcessing(result.process_id, this.selectedFiles);
            } else {
                throw new Error('No process ID received from server');
            }

        } catch (error) {
            Utils.logger.error('Retry failed:', error);
            this.handleProcessingError(error);
        }
    }
}

// Create global processing manager instance
const processingManager = new ProcessingManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ProcessingManager, processingManager };
}