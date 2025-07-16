/**
 * Chat Manager
 * Handles chat interface and message processing
 */

class ChatManager {
    constructor() {
        this.currentMode = CONFIG.CHAT.DEFAULT_MODE;
        this.messageHistory = [];
        this.isProcessing = false;
        this.typingIndicator = null;
        this.init();
    }

    init() {
        this.setupElements();
        this.attachEventListeners();
        this.setupModeSelector();
    }

    setupElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // Mode selector
        this.modeButtons = document.querySelectorAll('.mode-btn');
        
        // System stats
        this.statusDocs = document.getElementById('statusDocs');
        this.statusChunks = document.getElementById('statusChunks');
        this.statusEntities = document.getElementById('statusEntities');

        // Validate required elements
        if (!this.chatMessages || !this.chatInput || !this.sendBtn) {
            Utils.logger.error('Required chat elements not found');
        }
    }

    attachEventListeners() {
        // Send button click
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => {
                this.handleSendMessage();
            });
        }

        // Chat input events
        if (this.chatInput) {
            // Enter key to send (Shift+Enter for new line)
            this.chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSendMessage();
                }
            });

            // Auto-resize textarea
            this.chatInput.addEventListener('input', () => {
                this.autoResizeInput();
            });

            // Character limit warning
            this.chatInput.addEventListener('input', Utils.debounce(() => {
                this.checkMessageLength();
            }, 100));
        }
    }

    setupModeSelector() {
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.setMode(btn.dataset.mode);
            });
        });
    }

    /**
     * Set query mode
     */
    setMode(mode) {
        if (!QUERY_MODES[mode]) {
            Utils.logger.warn('Invalid query mode:', mode);
            return;
        }

        this.currentMode = mode;
        
        // Update UI
        this.modeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        Utils.logger.info('Query mode changed to:', mode);
        
        // Store preference
        Utils.storage.set('chat_mode', mode);
    }

    /**
     * Auto-resize chat input
     */
    autoResizeInput() {
        if (!this.chatInput) return;

        this.chatInput.style.height = 'auto';
        const newHeight = Math.min(this.chatInput.scrollHeight, 120);
        this.chatInput.style.height = newHeight + 'px';
    }

    /**
     * Check message length and show warning
     */
    checkMessageLength() {
        if (!this.chatInput) return;

        const length = this.chatInput.value.length;
        const maxLength = CONFIG.CHAT.MAX_MESSAGE_LENGTH;
        
        if (length > maxLength * 0.9) {
            // Show warning when approaching limit
            const remaining = maxLength - length;
            this.showLengthWarning(remaining);
        } else {
            this.hideLengthWarning();
        }
    }

    /**
     * Show length warning
     */
    showLengthWarning(remaining) {
        let warning = document.getElementById('lengthWarning');
        
        if (!warning) {
            warning = document.createElement('div');
            warning.id = 'lengthWarning';
            warning.className = 'length-warning';
            warning.style.cssText = `
                position: absolute;
                bottom: -25px;
                right: 10px;
                font-size: 0.8rem;
                color: ${remaining <= 0 ? '#e53e3e' : '#d69e2e'};
                background: white;
                padding: 2px 6px;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            `;
            
            this.chatInput.parentNode.style.position = 'relative';
            this.chatInput.parentNode.appendChild(warning);
        }
        
        warning.textContent = remaining <= 0 ? 'Message too long!' : `${remaining} characters remaining`;
        warning.style.color = remaining <= 0 ? '#e53e3e' : '#d69e2e';
    }

    /**
     * Hide length warning
     */
    hideLengthWarning() {
        const warning = document.getElementById('lengthWarning');
        if (warning) {
            warning.remove();
        }
    }

    /**
     * Handle send message
     */
    async handleSendMessage() {
        const message = this.chatInput.value.trim();
        
        if (!message || this.isProcessing) {
            return;
        }

        // Check message length
        if (message.length > CONFIG.CHAT.MAX_MESSAGE_LENGTH) {
            this.showError('Message is too long. Please shorten your message.');
            return;
        }

        Utils.logger.info('Sending message:', message);

        // Clear input and reset height
        this.chatInput.value = '';
        this.autoResizeInput();
        this.hideLengthWarning();

        // Add user message
        await this.addMessage(message, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        // Set processing state
        this.setProcessingState(true);

        try {
            // Send query to API
            const result = await apiClient.queryDocuments(message, {
                mode: this.currentMode,
                top_k: 20
            });

            // Remove typing indicator
            this.hideTypingIndicator();

            // Add assistant response
            await this.addMessage(result.response, 'assistant', {
                processingTime: result.processing_time,
                mode: result.mode,
                sources: result.sources_used
            });

            // Store in history
            this.addToHistory(message, result.response, this.currentMode);

        } catch (error) {
            Utils.logger.error('Query failed:', error);
            
            // Remove typing indicator
            this.hideTypingIndicator();
            
            // Show error message
            const errorMessage = this.getErrorMessage(error);
            await this.addMessage(errorMessage, 'assistant');
        } finally {
            // Reset processing state
            this.setProcessingState(false);
            
            // Focus input
            this.chatInput.focus();
        }
    }

    /**
     * Add message to chat
     */
    async addMessage(content, sender, meta = {}) {
        if (!this.chatMessages) return;

        const messageId = Utils.generateId('msg');
        const timestamp = Utils.formatTimestamp();
        const isUser = sender === 'user';

        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.id = messageId;

        const avatar = isUser ? 'You' : 'FA';
        const metaText = this.formatMessageMeta(meta, timestamp);

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${this.formatMessageContent(content)}
                <div class="message-meta">
                    <span>${metaText}</span>
                </div>
            </div>
        `;

        // Add to DOM
        this.chatMessages.appendChild(messageDiv);

        // Animate message appearance
        await animationManager.animateMessage(messageDiv, isUser);

        // Scroll to bottom
        this.scrollToBottom();

        // Add to history
        this.messageHistory.push({
            id: messageId,
            content,
            sender,
            timestamp: new Date(),
            meta
        });

        // Limit history size
        if (this.messageHistory.length > CONFIG.CHAT.MAX_HISTORY) {
            const removed = this.messageHistory.shift();
            const removedElement = document.getElementById(removed.id);
            if (removedElement) {
                removedElement.remove();
            }
        }
    }

    /**
     * Format message content
     */
    formatMessageContent(content) {
        // Parse simple markdown-like formatting
        return Utils.parseSimpleMarkdown(Utils.escapeHtml(content));
    }

    /**
     * Format message metadata
     */
    formatMessageMeta(meta, timestamp) {
        if (meta.processingTime) {
            const time = Utils.formatDuration(meta.processingTime);
            const mode = meta.mode || 'unknown';
            const sources = meta.sources || 0;
            return `${mode} mode • ${time} • ${sources} sources`;
        }
        
        if (meta.mode === 'system') {
            return meta.sources || 'System message';
        }
        
        return timestamp;
    }

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        if (this.typingIndicator) return;

        this.typingIndicator = document.createElement('div');
        this.typingIndicator.className = 'message assistant';
        this.typingIndicator.id = 'typing-indicator';
        
        this.typingIndicator.innerHTML = `
            <div class="message-avatar">FA</div>
            <div class="typing-indicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        this.chatMessages.appendChild(this.typingIndicator);
        this.scrollToBottom();
    }

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.remove();
            this.typingIndicator = null;
        }
    }

    /**
     * Set processing state
     */
    setProcessingState(processing) {
        this.isProcessing = processing;
        
        if (this.sendBtn) {
            this.sendBtn.disabled = processing;
        }
        
        if (this.chatInput) {
            this.chatInput.disabled = processing;
        }
    }

    /**
     * Scroll to bottom of messages
     */
    scrollToBottom(smooth = true) {
        if (!this.chatMessages) return;

        const scrollOptions = {
            top: this.chatMessages.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
        };

        this.chatMessages.scrollTo(scrollOptions);
    }

    /**
     * Add welcome message with processing stats
     */
    async addWelcomeMessage(processingStatus = {}) {
        const welcomeContent = `🎉 **System Ready!** I've successfully processed your documents and built a comprehensive knowledge graph.

**Processing Summary:**
• Documents processed: ${processingStatus.documents_processed || this.getFileCount()}
• Total entities extracted: ${processingStatus.entities_extracted || 'Multiple'}
• Relationships mapped: ${processingStatus.relationships_extracted || 'Multiple'}
• Processing time: ${processingStatus.processing_time ? Utils.formatDuration(processingStatus.processing_time) : 'Complete'}

You can now ask me questions about your financial data, and I'll provide detailed, context-aware responses using advanced retrieval techniques.

**Try asking:**
• "What are the key financial highlights?"
• "Compare revenue trends across quarters"
• "What are the main risk factors mentioned?"`;

        await this.addMessage(welcomeContent, 'assistant', {
            mode: 'system',
            sources: 'knowledge_graph'
        });
    }

    /**
     * Get file count from upload manager
     */
    getFileCount() {
        if (typeof uploadManager !== 'undefined') {
            return uploadManager.getSelectedFiles().length;
        }
        return 'N/A';
    }

    /**
     * Get error message for display
     */
    getErrorMessage(error) {
        if (error.status === 408) {
            return "I'm sorry, but your query timed out. This might be due to a complex question or high server load. Please try rephrasing your question or try again in a moment.";
        } else if (error.status >= 500) {
            return "I encountered a server error while processing your query. Please try again in a moment. If the problem persists, there might be an issue with the system.";
        } else if (error.status === 404) {
            return "I couldn't find the information you're looking for. Please make sure your documents have been processed successfully.";
        } else {
            return `I encountered an error while processing your query: ${error.message}. Please try rephrasing your question or try again.`;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        // You could implement a toast notification or inline error here
        Utils.logger.error('Chat error:', message);
    }

    /**
     * Add to message history
     */
    addToHistory(userMessage, assistantResponse, mode) {
        const historyEntry = {
            timestamp: new Date(),
            user: userMessage,
            assistant: assistantResponse,
            mode
        };

        // Store in local storage (limited history)
        const history = Utils.storage.get('chat_history', []);
        history.push(historyEntry);
        
        // Keep only last 20 exchanges
        if (history.length > 20) {
            history.shift();
        }
        
        Utils.storage.set('chat_history', history);
    }

    /**
     * Load chat history
     */
    loadChatHistory() {
        const history = Utils.storage.get('chat_history', []);
        
        // Load last few messages
        const recentHistory = history.slice(-5);
        
        recentHistory.forEach(async (entry) => {
            await this.addMessage(entry.user, 'user');
            await this.addMessage(entry.assistant, 'assistant', {
                mode: entry.mode,
                sources: 'cached'
            });
        });
    }

    /**
     * Clear chat history
     */
    clearHistory() {
        // Clear UI
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
        }
        
        // Clear storage
        Utils.storage.remove('chat_history');
        
        // Clear memory
        this.messageHistory = [];
        
        Utils.logger.info('Chat history cleared');
    }

    /**
     * Export chat history
     */
    exportHistory() {
        const history = this.messageHistory.map(msg => ({
            timestamp: msg.timestamp,
            sender: msg.sender,
            content: msg.content,
            meta: msg.meta
        }));

        const dataStr = JSON.stringify(history, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `chat-history-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }

    /**
     * Get current chat state
     */
    getState() {
        return {
            mode: this.currentMode,
            messageCount: this.messageHistory.length,
            isProcessing: this.isProcessing,
            lastMessage: this.messageHistory[this.messageHistory.length - 1]
        };
    }

    /**
     * Reset chat interface
     */
    reset() {
        this.clearHistory();
        this.setMode(CONFIG.CHAT.DEFAULT_MODE);
        this.setProcessingState(false);
        this.hideTypingIndicator();
        this.hideLengthWarning();
        
        if (this.chatInput) {
            this.chatInput.value = '';
            this.autoResizeInput();
        }
    }
}

// Create global chat manager instance
const chatManager = new ChatManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChatManager, chatManager };
}