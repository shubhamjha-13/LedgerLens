/**
 * Configuration and Constants
 * Central configuration for the Financial RAG system
 */

// API Configuration
const CONFIG = {
    // Backend API base URL
    API_BASE: 'http://localhost:8000',
    
    // Upload settings
    UPLOAD: {
        MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
        ALLOWED_TYPES: ['.pdf'],
        MAX_FILES: 10,
        CHUNK_SIZE: 1024 * 1024 // 1MB chunks for large files
    },
    
    // Processing settings
    PROCESSING: {
        POLL_INTERVAL: 1000, // 1 second
        MAX_POLL_ATTEMPTS: 300, // 5 minutes max
        TIMEOUT: 5 * 60 * 1000 // 5 minutes
    },
    
    // Chat settings
    CHAT: {
        MAX_HISTORY: 50,
        DEFAULT_MODE: 'hybrid',
        QUERY_TIMEOUT: 30000, // 30 seconds
        MAX_MESSAGE_LENGTH: 2000,
        TYPING_DELAY: 500
    },
    
    // Animation settings
    ANIMATION: {
        DURATION: {
            FAST: 300,
            NORMAL: 500,
            SLOW: 1000
        },
        EASING: 'cubic-bezier(0.4, 0, 0.2, 1)',
        PARTICLES: {
            COUNT: 20,
            LIFETIME: 8000,
            GENERATION_INTERVAL: 800
        }
    },
    
    // UI settings
    UI: {
        MOBILE_BREAKPOINT: 768,
        TABLET_BREAKPOINT: 1024,
        MAX_MODAL_WIDTH: 500,
        SIDEBAR_WIDTH: 300
    },
    
    // Debug settings
    DEBUG: {
        ENABLED: false,
        LOG_LEVEL: 'info', // error, warn, info, debug
        PERFORMANCE_MONITORING: false
    }
};

// Query modes configuration
const QUERY_MODES = {
    hybrid: {
        name: 'Hybrid',
        description: 'Combines local entity search with global themes',
        icon: '🔗',
        color: '#667eea'
    },
    local: {
        name: 'Local',
        description: 'Focuses on specific entities and relationships',
        icon: '🎯',
        color: '#48bb78'
    },
    global: {
        name: 'Global',
        description: 'Analyzes themes across all documents',
        icon: '🌐',
        color: '#ed8936'
    },
    mix: {
        name: 'Mix',
        description: 'Integrates vector search with knowledge graph',
        icon: '⚡',
        color: '#9f7aea'
    },
    naive: {
        name: 'Vector',
        description: 'Simple semantic similarity search',
        icon: '🔍',
        color: '#38b2ac'
    }
};

// File type validation
const FILE_VALIDATION = {
    pdf: {
        mimeTypes: ['application/pdf'],
        extensions: ['.pdf'],
        maxSize: CONFIG.UPLOAD.MAX_FILE_SIZE,
        icon: '📄'
    }
};

// Error messages
const ERROR_MESSAGES = {
    NETWORK: 'Network error. Please check your connection.',
    TIMEOUT: 'Request timed out. Please try again.',
    FILE_TOO_LARGE: 'File is too large. Maximum size is 50MB.',
    INVALID_FILE_TYPE: 'Invalid file type. Only PDF files are supported.',
    TOO_MANY_FILES: 'Too many files. Maximum is 10 files.',
    UPLOAD_FAILED: 'Upload failed. Please try again.',
    PROCESSING_FAILED: 'Processing failed. Please check your documents.',
    QUERY_FAILED: 'Query failed. Please try again.',
    SYSTEM_ERROR: 'System error. Please contact support.',
    NO_API_KEY: 'OpenAI API key not configured.',
    SERVER_UNAVAILABLE: 'Server is unavailable. Please try again later.'
};

// Success messages
const SUCCESS_MESSAGES = {
    UPLOAD_COMPLETE: 'Documents uploaded successfully!',
    PROCESSING_COMPLETE: 'Your Financial RAG system is ready!',
    QUERY_COMPLETE: 'Query processed successfully.',
    SYSTEM_READY: 'System is ready for queries.'
};

// UI Constants
const UI_CONSTANTS = {
    STAGES: {
        UPLOAD: 'upload',
        PROCESSING: 'processing',
        CHAT: 'chat'
    },
    
    THEMES: {
        LIGHT: 'light',
        DARK: 'dark',
        AUTO: 'auto'
    },
    
    BREAKPOINTS: {
        MOBILE: '(max-width: 768px)',
        TABLET: '(max-width: 1024px)',
        DESKTOP: '(min-width: 1025px)'
    }
};

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        QUERY_MODES,
        FILE_VALIDATION,
        ERROR_MESSAGES,
        SUCCESS_MESSAGES,
        UI_CONSTANTS
    };
}