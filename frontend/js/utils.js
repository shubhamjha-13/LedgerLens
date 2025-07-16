/**
 * Utility Functions
 * Common helper functions used throughout the application
 */

class Utils {
    /**
     * Format file size in human readable format
     */
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Format duration in human readable format
     */
    static formatDuration(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${minutes}m ${secs}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Format timestamp
     */
    static formatTimestamp(date = new Date()) {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    /**
     * Debounce function calls
     */
    static debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    }

    /**
     * Throttle function calls
     */
    static throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Generate unique ID
     */
    static generateId(prefix = 'id') {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Validate file type
     */
    static validateFile(file) {
        const errors = [];
        
        // Check file type
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            errors.push(ERROR_MESSAGES.INVALID_FILE_TYPE);
        }
        
        // Check file size
        if (file.size > CONFIG.UPLOAD.MAX_FILE_SIZE) {
            errors.push(ERROR_MESSAGES.FILE_TOO_LARGE);
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Validate multiple files
     */
    static validateFiles(files) {
        const fileArray = Array.from(files);
        const errors = [];
        
        // Check file count
        if (fileArray.length > CONFIG.UPLOAD.MAX_FILES) {
            errors.push(ERROR_MESSAGES.TOO_MANY_FILES);
        }
        
        // Check each file
        const fileErrors = [];
        fileArray.forEach((file, index) => {
            const validation = this.validateFile(file);
            if (!validation.valid) {
                fileErrors.push({
                    file: file.name,
                    index,
                    errors: validation.errors
                });
            }
        });
        
        return {
            valid: errors.length === 0 && fileErrors.length === 0,
            errors,
            fileErrors,
            validFiles: fileArray.filter((_, index) => 
                !fileErrors.some(fe => fe.index === index)
            )
        };
    }

    /**
     * Copy text to clipboard
     */
    static async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                return true;
            } catch (err) {
                return false;
            } finally {
                document.body.removeChild(textArea);
            }
        }
    }

    /**
     * Escape HTML
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Parse markdown-like formatting
     */
    static parseSimpleMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    /**
     * Truncate text
     */
    static truncateText(text, maxLength, suffix = '...') {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - suffix.length) + suffix;
    }

    /**
     * Check if device is mobile
     */
    static isMobile() {
        return window.innerWidth <= CONFIG.UI.MOBILE_BREAKPOINT;
    }

    /**
     * Check if device is tablet
     */
    static isTablet() {
        return window.innerWidth <= CONFIG.UI.TABLET_BREAKPOINT && 
               window.innerWidth > CONFIG.UI.MOBILE_BREAKPOINT;
    }

    /**
     * Get device type
     */
    static getDeviceType() {
        if (this.isMobile()) return 'mobile';
        if (this.isTablet()) return 'tablet';
        return 'desktop';
    }

    /**
     * Smooth scroll to element
     */
    static scrollToElement(element, options = {}) {
        const defaultOptions = {
            behavior: 'smooth',
            block: 'start',
            inline: 'nearest'
        };
        
        element.scrollIntoView({ ...defaultOptions, ...options });
    }

    /**
     * Wait for specified time
     */
    static wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Retry function with exponential backoff
     */
    static async retry(fn, options = {}) {
        const {
            maxAttempts = 3,
            baseDelay = 1000,
            maxDelay = 10000,
            backoffFactor = 2
        } = options;

        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (attempt === maxAttempts) {
                    throw lastError;
                }
                
                const delay = Math.min(
                    baseDelay * Math.pow(backoffFactor, attempt - 1),
                    maxDelay
                );
                
                await this.wait(delay);
            }
        }
    }

    /**
     * Deep clone object
     */
    static deepClone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => this.deepClone(item));
        if (typeof obj === 'object') {
            const cloned = {};
            Object.keys(obj).forEach(key => {
                cloned[key] = this.deepClone(obj[key]);
            });
            return cloned;
        }
    }

    /**
     * Check if object is empty
     */
    static isEmpty(obj) {
        if (obj == null) return true;
        if (Array.isArray(obj) || typeof obj === 'string') return obj.length === 0;
        return Object.keys(obj).length === 0;
    }

    /**
     * Merge objects deeply
     */
    static deepMerge(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source.hasOwnProperty(key)) {
                if (this.isObject(source[key]) && this.isObject(target[key])) {
                    result[key] = this.deepMerge(target[key], source[key]);
                } else {
                    result[key] = source[key];
                }
            }
        }
        
        return result;
    }

    /**
     * Check if value is object
     */
    static isObject(item) {
        return item && typeof item === 'object' && !Array.isArray(item);
    }

    /**
     * Local storage helpers
     */
    static storage = {
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.warn('Failed to save to localStorage:', error);
                return false;
            }
        },

        get(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.warn('Failed to read from localStorage:', error);
                return defaultValue;
            }
        },

        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.warn('Failed to remove from localStorage:', error);
                return false;
            }
        },

        clear() {
            try {
                localStorage.clear();
                return true;
            } catch (error) {
                console.warn('Failed to clear localStorage:', error);
                return false;
            }
        }
    };

    /**
     * Logger utility
     */
    static logger = {
        error(...args) {
            if (CONFIG.DEBUG.LOG_LEVEL === 'error' || CONFIG.DEBUG.ENABLED) {
                console.error('[RAG]', ...args);
            }
        },

        warn(...args) {
            if (['error', 'warn'].includes(CONFIG.DEBUG.LOG_LEVEL) || CONFIG.DEBUG.ENABLED) {
                console.warn('[RAG]', ...args);
            }
        },

        info(...args) {
            if (['error', 'warn', 'info'].includes(CONFIG.DEBUG.LOG_LEVEL) || CONFIG.DEBUG.ENABLED) {
                console.info('[RAG]', ...args);
            }
        },

        debug(...args) {
            if (CONFIG.DEBUG.ENABLED) {
                console.debug('[RAG]', ...args);
            }
        }
    };
}

// Export Utils
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}