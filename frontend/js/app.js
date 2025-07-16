/**
 * Main Application Controller
 * Orchestrates all components and manages global application state
 */

class FinancialRAGApp {
    constructor() {
        this.currentStage = UI_CONSTANTS.STAGES.UPLOAD;
        this.systemReady = false;
        this.initialized = false;
        this.healthCheckInterval = null;
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        Utils.logger.info('Initializing Financial RAG Application');

        try {
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve);
                });
            }

            // Initialize components
            await this.initializeComponents();

            // Perform health check
            await this.performHealthCheck();

            // Setup global event listeners
            this.setupGlobalEventListeners();

            // Load user preferences
            this.loadUserPreferences();

            // Mark as initialized
            this.initialized = true;

            Utils.logger.info('Application initialized successfully');

        } catch (error) {
            Utils.logger.error('Application initialization failed:', error);
            this.handleInitializationError(error);
        }
    }

    /**
     * Initialize all components
     */
    async initializeComponents() {
        Utils.logger.info('Initializing components...');

        // Components should already be initialized via their constructors
        // This method can be used for additional setup if needed

        // Validate that all managers are available
        const requiredManagers = [
            'uploadManager',
            'processingManager', 
            'chatManager',
            'animationManager',
            'apiClient'
        ];

        const missingManagers = requiredManagers.filter(manager => 
            typeof window[manager] === 'undefined'
        );

        if (missingManagers.length > 0) {
            throw new Error(`Missing required managers: ${missingManagers.join(', ')}`);
        }

        Utils.logger.info('All components initialized');
    }

    /**
     * Perform initial health check
     */
    async performHealthCheck() {
        Utils.logger.info('Performing health check...');

        try {
            const health = await apiClient.checkHealth();
            
            if (health.healthy) {
                this.systemReady = health.data.rag_system_ready;
                Utils.logger.info('System health check passed', health.data);
            } else {
                Utils.logger.warn('System health check failed', health.error);
                this.showSystemWarning('Backend system is not responding. Please ensure the server is running.');
            }

        } catch (error) {
            Utils.logger.error('Health check failed:', error);
            this.showSystemWarning('Cannot connect to backend. Please check your connection and ensure the server is running.');
        }
    }

    /**
     * Setup global event listeners
     */
    setupGlobalEventListeners() {
        // Handle browser back/forward buttons
        window.addEventListener('popstate', (event) => {
            if (event.state && event.state.stage) {
                this.navigateToStage(event.state.stage, false);
            }
        });

        // Handle window resize
        window.addEventListener('resize', Utils.throttle(() => {
            this.handleWindowResize();
        }, 100));

        // Handle visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            this.handleVisibilityChange();
        });

        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleGlobalKeyboard(e);
        });

        // Handle unload (page refresh/close)
        window.addEventListener('beforeunload', (e) => {
            this.handleBeforeUnload(e);
        });

        // Global error handler
        window.addEventListener('error', (event) => {
            Utils.logger.error('Global error:', event.error);
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            Utils.logger.error('Unhandled promise rejection:', event.reason);
        });
    }

    /**
     * Load user preferences
     */
    loadUserPreferences() {
        // Load saved query mode
        const savedMode = Utils.storage.get('chat_mode');
        if (savedMode && QUERY_MODES[savedMode]) {
            chatManager.setMode(savedMode);
        }

        // Load theme preference (if implemented)
        const savedTheme = Utils.storage.get('theme', 'light');
        this.setTheme(savedTheme);

        // Load other preferences as needed
        Utils.logger.info('User preferences loaded');
    }

    /**
     * Navigate to specific stage
     */
    async navigateToStage(stage, updateHistory = true) {
        if (!this.initialized) {
            Utils.logger.warn('Cannot navigate - app not initialized');
            return;
        }

        const stages = [
            UI_CONSTANTS.STAGES.UPLOAD,
            UI_CONSTANTS.STAGES.PROCESSING, 
            UI_CONSTANTS.STAGES.CHAT
        ];

        if (!stages.includes(stage)) {
            Utils.logger.error('Invalid stage:', stage);
            return;
        }

        Utils.logger.info(`Navigating to stage: ${stage}`);

        const fromStage = this.currentStage;
        this.currentStage = stage;

        // Update browser history
        if (updateHistory) {
            const url = new URL(window.location);
            url.searchParams.set('stage', stage);
            history.pushState({ stage }, '', url);
        }

        // Perform stage-specific transitions
        await this.performStageTransition(fromStage, stage);
    }

    /**
     * Perform stage transition
     */
    async performStageTransition(fromStage, toStage) {
        const stageElements = {
            [UI_CONSTANTS.STAGES.UPLOAD]: document.getElementById('uploadStage'),
            [UI_CONSTANTS.STAGES.PROCESSING]: document.getElementById('processingStage'),
            [UI_CONSTANTS.STAGES.CHAT]: document.getElementById('chatStage')
        };

        const fromElement = stageElements[fromStage];
        const toElement = stageElements[toStage];

        if (!fromElement || !toElement) {
            Utils.logger.error('Stage elements not found');
            return;
        }

        // Use animation manager for smooth transitions
        await animationManager.transitionStage(fromElement, toElement);

        // Stage-specific logic
        switch (toStage) {
            case UI_CONSTANTS.STAGES.UPLOAD:
                uploadManager.reset();
                break;
                
            case UI_CONSTANTS.STAGES.CHAT:
                this.loadSystemStats();
                // Focus chat input after transition
                setTimeout(() => {
                    const chatInput = document.getElementById('chatInput');
                    if (chatInput) chatInput.focus();
                }, 100);
                break;
        }
    }

    /**
     * Load and display system statistics
     */
    async loadSystemStats() {
        try {
            const stats = await apiClient.getSystemStatus();
            this.updateSystemStats(stats);
        } catch (error) {
            Utils.logger.error('Failed to load system stats:', error);
        }
    }

    /**
     * Update system statistics display
     */
    updateSystemStats(stats) {
        const elements = {
            statusDocs: document.getElementById('statusDocs'),
            statusChunks: document.getElementById('statusChunks'),
            statusEntities: document.getElementById('statusEntities')
        };

        if (elements.statusDocs) {
            elements.statusDocs.textContent = `${stats.total_documents || 0} docs`;
        }
        
        if (elements.statusChunks) {
            elements.statusChunks.textContent = `${stats.total_chunks || 0} chunks`;
        }
        
        if (elements.statusEntities) {
            elements.statusEntities.textContent = `${stats.total_entities || 0} entities`;
        }
    }

    /**
     * Handle window resize
     */
    handleWindowResize() {
        // Update mobile/desktop specific logic
        const deviceType = Utils.getDeviceType();
        document.body.setAttribute('data-device-type', deviceType);

        // Auto-resize chat input if visible
        if (this.currentStage === UI_CONSTANTS.STAGES.CHAT && chatManager.chatInput) {
            chatManager.autoResizeInput();
        }
    }

    /**
     * Handle visibility change (tab switching)
     */
    handleVisibilityChange() {
        if (document.hidden) {
            // Tab is hidden - pause non-essential animations
            if (processingManager.isProcessing()) {
                // Keep processing monitoring active
            } else {
                // Pause decorative animations
                animationManager.stopAllParticles();
            }
        } else {
            // Tab is visible - resume animations if needed
            if (processingManager.isProcessing() && processingManager.particleAnimation) {
                // Particles will auto-resume via processing manager
            }
        }
    }

    /**
     * Handle global keyboard shortcuts
     */
    handleGlobalKeyboard(event) {
        // Ctrl/Cmd + /  - Show help (if implemented)
        if ((event.ctrlKey || event.metaKey) && event.key === '/') {
            event.preventDefault();
            this.showHelp();
        }

        // Escape key - Cancel operations
        if (event.key === 'Escape') {
            this.handleEscapeKey();
        }

        // Ctrl/Cmd + K - Focus search/chat input
        if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
            event.preventDefault();
            this.focusChatInput();
        }
    }

    /**
     * Handle before unload (page refresh/close)
     */
    handleBeforeUnload(event) {
        // Warn if processing is in progress
        if (processingManager.isProcessing()) {
            event.preventDefault();
            event.returnValue = 'Document processing is in progress. Are you sure you want to leave?';
            return event.returnValue;
        }

        // Clean up resources
        this.cleanup();
    }

    /**
     * Handle escape key
     */
    handleEscapeKey() {
        // Close modals or cancel operations
        if (processingManager.isProcessing()) {
            // Could implement cancel processing here
            Utils.logger.info('Escape pressed during processing');
        }

        // Clear any error messages
        this.clearErrorMessages();
    }

    /**
     * Focus chat input
     */
    focusChatInput() {
        if (this.currentStage === UI_CONSTANTS.STAGES.CHAT && chatManager.chatInput) {
            chatManager.chatInput.focus();
        }
    }

    /**
     * Show help modal (if implemented)
     */
    showHelp() {
        // Could implement help modal here
        Utils.logger.info('Help requested');
    }

    /**
     * Show system warning
     */
    showSystemWarning(message) {
        // Create or update system warning banner
        let warning = document.getElementById('systemWarning');
        
        if (!warning) {
            warning = document.createElement('div');
            warning.id = 'systemWarning';
            warning.className = 'system-warning';
            warning.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: #fed7d7;
                color: #e53e3e;
                padding: 12px 20px;
                text-align: center;
                font-size: 0.9rem;
                font-weight: 500;
                z-index: 9999;
                border-bottom: 1px solid #feb2b2;
            `;
            
            document.body.prepend(warning);
        }
        
        warning.textContent = message;
        warning.style.display = 'block';
    }

    /**
     * Hide system warning
     */
    hideSystemWarning() {
        const warning = document.getElementById('systemWarning');
        if (warning) {
            warning.style.display = 'none';
        }
    }

    /**
     * Clear error messages
     */
    clearErrorMessages() {
        // Clear upload errors
        const uploadError = document.getElementById('uploadError');
        if (uploadError) {
            uploadError.classList.add('hidden');
        }

        // Clear processing errors
        const processingError = document.getElementById('processingModalError');
        if (processingError) {
            processingError.classList.add('hidden');
        }
    }

    /**
     * Set application theme
     */
    setTheme(theme) {
        document.body.setAttribute('data-theme', theme);
        Utils.storage.set('theme', theme);
    }

    /**
     * Start periodic health checks
     */
    startHealthChecks() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        // Check health every 5 minutes
        this.healthCheckInterval = setInterval(async () => {
            try {
                const health = await apiClient.checkHealth();
                
                if (!health.healthy && this.systemReady) {
                    this.showSystemWarning('Connection to backend lost. Please check your connection.');
                    this.systemReady = false;
                } else if (health.healthy && !this.systemReady) {
                    this.hideSystemWarning();
                    this.systemReady = true;
                }
            } catch (error) {
                Utils.logger.debug('Health check failed:', error);
            }
        }, 5 * 60 * 1000);
    }

    /**
     * Stop periodic health checks
     */
    stopHealthChecks() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }

    /**
     * Handle initialization error
     */
    handleInitializationError(error) {
        // Show error message to user
        const errorMessage = `
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                text-align: center;
                max-width: 400px;
                z-index: 10000;
            ">
                <h2 style="color: #e53e3e; margin-bottom: 20px;">Initialization Error</h2>
                <p style="color: #4a5568; margin-bottom: 20px;">
                    The application failed to initialize properly. Please refresh the page and try again.
                </p>
                <button onclick="window.location.reload()" style="
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 1rem;
                ">
                    Refresh Page
                </button>
            </div>
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.8);
                z-index: 9999;
            "></div>
        `;
        
        document.body.innerHTML = errorMessage;
    }

    /**
     * Clean up resources
     */
    cleanup() {
        Utils.logger.info('Cleaning up application resources');

        // Stop health checks
        this.stopHealthChecks();

        // Stop animations
        animationManager.stopAllAnimations();

        // Clear any intervals/timeouts
        // (Individual managers should handle their own cleanup)

        // Save any pending data
        // (Handled by individual managers)
    }

    /**
     * Get application state
     */
    getState() {
        return {
            currentStage: this.currentStage,
            systemReady: this.systemReady,
            initialized: this.initialized,
            upload: uploadManager.getSelectedFiles().length > 0,
            processing: processingManager.isProcessing(),
            chat: chatManager.getState()
        };
    }

    /**
     * Reset application to initial state
     */
    async reset() {
        Utils.logger.info('Resetting application');

        // Reset all managers
        uploadManager.reset();
        chatManager.reset();
        
        // Cancel any ongoing processing
        if (processingManager.isProcessing()) {
            await processingManager.cancelProcessing();
        }

        // Navigate to upload stage
        await this.navigateToStage(UI_CONSTANTS.STAGES.UPLOAD);

        // Clear system warnings
        this.hideSystemWarning();
        this.clearErrorMessages();
    }
}

// Initialize application when DOM is ready
let app;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        app = new FinancialRAGApp();
    });
} else {
    app = new FinancialRAGApp();
}

// Make app available globally for debugging
window.ragApp = app;

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FinancialRAGApp };
}