/* =================================================================
   Processing Modal - Beautiful processing interface and animations
   ================================================================= */

/* Processing Overlay */
.processing-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(10px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    opacity: 0;
    visibility: hidden;
    transition: all 0.5s ease;
}

.processing-overlay.active {
    opacity: 1;
    visibility: visible;
}

/* Processing Modal */
.processing-modal {
    background: white;
    border-radius: 24px;
    padding: 60px 50px;
    text-align: center;
    box-shadow: 0 30px 60px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.2);
    max-width: 500px;
    width: 90%;
    transform: scale(0.8) translateY(20px);
    transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}

.processing-overlay.active .processing-modal {
    transform: scale(1) translateY(0);
}

/* Processing Animation */
.processing-animation {
    width: 140px;
    height: 140px;
    margin: 0 auto 40px;
    position: relative;
}

.processing-circle {
    width: 100%;
    height: 100%;
    border: 6px solid rgba(102, 126, 234, 0.1);
    border-radius: 50%;
    position: relative;
    overflow: hidden;
}

.processing-circle::before {
    content: '';
    position: absolute;
    top: -6px;
    left: -6px;
    width: 100%;
    height: 100%;
    border: 6px solid transparent;
    border-top: 6px solid #667eea;
    border-right: 6px solid #667eea;
    border-radius: 50%;
    animation: spin 2s linear infinite;
}

.processing-circle::after {
    content: '';
    position: absolute;
    top: 20px;
    left: 20px;
    right: 20px;
    bottom: 20px;
    border: 3px solid rgba(102, 126, 234, 0.2);
    border-top: 3px solid #764ba2;
    border-radius: 50%;
    animation: spin 1.5s linear infinite reverse;
}

.processing-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: heartbeat 2s ease-in-out infinite;
}

.processing-icon {
    width: 30px;
    height: 30px;
    fill: white;
}

/* Processing Text */
.processing-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 20px;
    animation: titleGlow 3s ease-in-out infinite;
}

.processing-message {
    color: #4a5568;
    font-size: 1.2rem;
    margin-bottom: 15px;
    font-weight: 500;
    animation: fadeInOut 2s ease-in-out infinite;
}

.processing-submessage {
    color: #718096;
    font-size: 1rem;
    margin-bottom: 40px;
    animation: slideUp 0.5s ease-out;
}

/* Document List */
.document-list {
    background: #f8fafc;
    border-radius: 16px;
    padding: 20px;
    margin: 30px 0;
    border-left: 4px solid #667eea;
    text-align: left;
    max-height: 200px;
    overflow-y: auto;
}

.document-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #e2e8f0;
    animation: documentSlide 0.5s ease-out;
    transition: all 0.3s ease;
}

.document-item:last-child {
    border-bottom: none;
}

.document-item.processing {
    background: rgba(102, 126, 234, 0.1);
    border-radius: 8px;
    padding: 12px;
    margin: 4px 0;
    border-bottom: none;
}

.document-status {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
    transition: all 0.3s ease;
}

.document-status.pending {
    background: #e2e8f0;
}

.document-status.processing {
    background: #667eea;
    animation: pulse 1s ease-in-out infinite;
}

.document-status.completed {
    background: #48bb78;
    animation: completedPulse 0.5s ease-out;
}

.document-status.error {
    background: #e53e3e;
}

.document-name {
    font-size: 0.9rem;
    color: #2d3748;
    font-weight: 500;
}

/* Progress Container */
.progress-container {
    margin: 30px 0;
}

.progress-bar {
    width: 100%;
    height: 12px;
    background: #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    position: relative;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    width: 0%;
    position: relative;
    overflow: hidden;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.4),
        transparent
    );
    animation: shimmer 2s infinite;
}

.progress-text {
    color: #4a5568;
    font-size: 1rem;
    margin-top: 15px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.progress-details {
    font-size: 0.85rem;
    color: #718096;
    font-weight: 400;
}

/* Floating Particles */
.floating-particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: #667eea;
    border-radius: 50%;
    animation: float 6s infinite linear;
    opacity: 0.6;
}

.particle:nth-child(odd) {
    background: #764ba2;
}

.particle:nth-child(even) {
    background: #667eea;
}

/* Celebration Effect */
.celebration {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 3rem;
    animation: celebrate 0.8s ease-out;
    pointer-events: none;
    z-index: 10;
}

/* Legacy Processing Stage (fallback) */
.processing-stage {
    background: white;
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    border: 1px solid rgba(0,0,0,0.05);
}

.processing-dots {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

.dot {
    width: 8px;
    height: 8px;
    background: #667eea;
    border-radius: 50%;
    margin: 0 4px;
    animation: pulse 1.4s ease-in-out infinite both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
.dot:nth-child(3) { animation-delay: 0; }

/* Mobile Responsive */
@media (max-width: 768px) {
    .processing-modal {
        padding: 40px 30px;
        margin: 20px;
    }

    .processing-animation {
        width: 100px;
        height: 100px;
        margin-bottom: 30px;
    }

    .processing-title {
        font-size: 1.8rem;
    }

    .processing-message {
        font-size: 1.1rem;
    }

    .document-list {
        padding: 15px;
        margin: 20px 0;
    }
}

@media (max-width: 480px) {
    .processing-modal {
        padding: 30px 20px;
        border-radius: 16px;
    }

    .processing-animation {
        width: 80px;
        height: 80px;
    }

    .processing-title {
        font-size: 1.5rem;
    }

    .processing-message {
        font-size: 1rem;
    }

    .processing-submessage {
        font-size: 0.9rem;
    }
}