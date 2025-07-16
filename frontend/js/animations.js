/**
 * Animation Manager
 * Handles all animations and visual effects
 */

class AnimationManager {
    constructor() {
        this.activeAnimations = new Map();
        this.particleIntervals = new Set();
        this.init();
    }

    init() {
        // Check for reduced motion preference
        this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        
        // Listen for reduced motion changes
        window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', (e) => {
            this.prefersReducedMotion = e.matches;
            if (this.prefersReducedMotion) {
                this.stopAllAnimations();
            }
        });
    }

    /**
     * Create floating particles animation
     */
    createFloatingParticles(container, options = {}) {
        if (this.prefersReducedMotion) return null;

        const {
            count = CONFIG.ANIMATION.PARTICLES.COUNT,
            lifetime = CONFIG.ANIMATION.PARTICLES.LIFETIME,
            interval = CONFIG.ANIMATION.PARTICLES.GENERATION_INTERVAL,
            colors = ['#667eea', '#764ba2']
        } = options;

        let particleCount = 0;
        
        const generateParticle = () => {
            if (!container || particleCount >= count * 2) return;

            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = '0s';
            particle.style.animationDuration = (4 + Math.random() * 4) + 's';
            particle.style.background = colors[Math.floor(Math.random() * colors.length)];
            
            container.appendChild(particle);
            particleCount++;
            
            // Remove particle after animation
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.parentNode.removeChild(particle);
                    particleCount--;
                }
            }, lifetime);
        };
        
        // Initial burst
        for (let i = 0; i < count; i++) {
            setTimeout(generateParticle, i * 100);
        }
        
        // Continuous generation
        const particleInterval = setInterval(generateParticle, interval);
        this.particleIntervals.add(particleInterval);
        
        return {
            stop: () => {
                clearInterval(particleInterval);
                this.particleIntervals.delete(particleInterval);
            }
        };
    }

    /**
     * Stop all particle animations
     */
    stopAllParticles() {
        this.particleIntervals.forEach(interval => {
            clearInterval(interval);
        });
        this.particleIntervals.clear();
    }

    /**
     * Animate element entrance
     */
    animateIn(element, animation = 'fadeIn', duration = CONFIG.ANIMATION.DURATION.NORMAL) {
        if (this.prefersReducedMotion) {
            element.style.opacity = '1';
            element.style.transform = 'none';
            return Promise.resolve();
        }

        return new Promise((resolve) => {
            element.style.opacity = '0';
            element.style.transform = this.getInitialTransform(animation);
            element.style.transition = `all ${duration}ms ${CONFIG.ANIMATION.EASING}`;
            
            // Force reflow
            element.offsetHeight;
            
            // Apply final state
            element.style.opacity = '1';
            element.style.transform = 'none';
            
            setTimeout(resolve, duration);
        });
    }

    /**
     * Animate element exit
     */
    animateOut(element, animation = 'fadeOut', duration = CONFIG.ANIMATION.DURATION.NORMAL) {
        if (this.prefersReducedMotion) {
            element.style.opacity = '0';
            return Promise.resolve();
        }

        return new Promise((resolve) => {
            element.style.transition = `all ${duration}ms ${CONFIG.ANIMATION.EASING}`;
            element.style.opacity = '0';
            element.style.transform = this.getFinalTransform(animation);
            
            setTimeout(resolve, duration);
        });
    }

    /**
     * Get initial transform for animation type
     */
    getInitialTransform(animation) {
        const transforms = {
            fadeIn: 'translateY(20px)',
            slideIn: 'translateY(20px)',
            slideInLeft: 'translateX(-20px)',
            slideInRight: 'translateX(20px)',
            slideInUp: 'translateY(20px)',
            slideInDown: 'translateY(-20px)',
            zoomIn: 'scale(0.8)',
            bounceIn: 'scale(0.3)'
        };
        
        return transforms[animation] || 'translateY(20px)';
    }

    /**
     * Get final transform for animation type
     */
    getFinalTransform(animation) {
        const transforms = {
            fadeOut: 'translateY(-20px)',
            slideOut: 'translateY(-20px)',
            slideOutLeft: 'translateX(-100%)',
            slideOutRight: 'translateX(100%)',
            slideOutUp: 'translateY(-100%)',
            slideOutDown: 'translateY(100%)',
            zoomOut: 'scale(0.8)',
            bounceOut: 'scale(0.3)'
        };
        
        return transforms[animation] || 'translateY(-20px)';
    }

    /**
     * Stage transition animation
     */
    async transitionStage(fromStage, toStage, direction = 'forward') {
        if (this.prefersReducedMotion) {
            fromStage.classList.add('hidden');
            toStage.classList.remove('hidden');
            toStage.classList.add('active');
            return;
        }

        const duration = CONFIG.ANIMATION.DURATION.SLOW;
        
        // Animate out current stage
        await this.animateOut(fromStage, direction === 'forward' ? 'slideOutLeft' : 'slideOutRight', duration);
        fromStage.classList.add('hidden');
        fromStage.classList.remove('active');
        
        // Prepare new stage
        toStage.classList.remove('hidden');
        
        // Small delay for better visual separation
        await Utils.wait(100);
        
        // Animate in new stage
        await this.animateIn(toStage, direction === 'forward' ? 'slideInRight' : 'slideInLeft', duration);
        toStage.classList.add('active');
    }

    /**
     * Show processing modal with animation
     */
    async showProcessingModal(modal) {
        if (this.prefersReducedMotion) {
            modal.classList.add('active');
            return;
        }

        modal.style.opacity = '0';
        modal.style.visibility = 'visible';
        modal.querySelector('.processing-modal').style.transform = 'scale(0.8) translateY(20px)';
        
        // Force reflow
        modal.offsetHeight;
        
        // Animate in
        modal.style.transition = 'all 0.5s ease';
        modal.style.opacity = '1';
        
        const modalElement = modal.querySelector('.processing-modal');
        modalElement.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
        modalElement.style.transform = 'scale(1) translateY(0)';
        
        await Utils.wait(500);
    }

    /**
     * Hide processing modal with animation
     */
    async hideProcessingModal(modal) {
        if (this.prefersReducedMotion) {
            modal.classList.remove('active');
            return;
        }

        modal.style.transition = 'all 0.8s ease';
        modal.style.opacity = '0';
        modal.style.transform = 'scale(0.9)';
        
        await Utils.wait(800);
        
        modal.classList.remove('active');
        modal.style.visibility = 'hidden';
        modal.style.transition = '';
        modal.style.opacity = '';
        modal.style.transform = '';
        
        const modalElement = modal.querySelector('.processing-modal');
        modalElement.style.transition = '';
        modalElement.style.transform = '';
    }

    /**
     * Animate progress bar
     */
    animateProgress(progressBar, targetWidth, duration = 1000) {
        if (this.prefersReducedMotion) {
            progressBar.style.width = targetWidth + '%';
            return Promise.resolve();
        }

        return new Promise((resolve) => {
            progressBar.style.transition = `width ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
            progressBar.style.width = targetWidth + '%';
            
            setTimeout(resolve, duration);
        });
    }

    /**
     * Create success celebration effect
     */
    createCelebration(container, options = {}) {
        if (this.prefersReducedMotion) return;

        const {
            emoji = '🎉',
            duration = 800,
            size = '3rem'
        } = options;

        const celebration = document.createElement('div');
        celebration.className = 'celebration';
        celebration.textContent = emoji;
        celebration.style.fontSize = size;
        celebration.style.position = 'absolute';
        celebration.style.top = '50%';
        celebration.style.left = '50%';
        celebration.style.transform = 'translate(-50%, -50%)';
        celebration.style.pointerEvents = 'none';
        celebration.style.zIndex = '10';
        celebration.style.animation = `celebrate ${duration}ms ease-out`;
        
        container.appendChild(celebration);
        
        setTimeout(() => {
            if (celebration.parentNode) {
                celebration.parentNode.removeChild(celebration);
            }
        }, duration);
    }

    /**
     * Animate document status changes
     */
    animateDocumentStatus(element, status) {
        if (this.prefersReducedMotion) return;

        const statusElement = element.querySelector('.document-status');
        if (!statusElement) return;

        // Add status-specific animation
        switch (status) {
            case 'processing':
                statusElement.style.animation = 'pulse 1s ease-in-out infinite';
                element.classList.add('processing');
                break;
            case 'completed':
                statusElement.style.animation = 'completedPulse 0.5s ease-out';
                element.classList.remove('processing');
                setTimeout(() => {
                    statusElement.style.animation = '';
                }, 500);
                break;
            case 'error':
                statusElement.style.animation = 'shake 0.5s ease';
                element.classList.remove('processing');
                setTimeout(() => {
                    statusElement.style.animation = '';
                }, 500);
                break;
            default:
                statusElement.style.animation = '';
                element.classList.remove('processing');
        }
    }

    /**
     * Animate message appearance in chat
     */
    animateMessage(messageElement, isUser = false) {
        if (this.prefersReducedMotion) {
            messageElement.style.opacity = '1';
            messageElement.style.transform = 'none';
            return Promise.resolve();
        }

        const direction = isUser ? 'translateX(20px)' : 'translateX(-20px)';
        
        messageElement.style.opacity = '0';
        messageElement.style.transform = `${direction} translateY(10px)`;
        messageElement.style.transition = 'all 0.3s ease-out';
        
        // Force reflow
        messageElement.offsetHeight;
        
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'none';
        
        return Utils.wait(300);
    }

    /**
     * Animate typing indicator
     */
    createTypingIndicator(container) {
        if (this.prefersReducedMotion) {
            const simple = document.createElement('div');
            simple.textContent = 'Typing...';
            simple.style.color = '#a0aec0';
            simple.style.fontSize = '0.9rem';
            simple.style.padding = '16px 20px';
            container.appendChild(simple);
            return simple;
        }

        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = `
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        container.appendChild(typingIndicator);
        return typingIndicator;
    }

    /**
     * Button press animation
     */
    animateButtonPress(button) {
        if (this.prefersReducedMotion) return;

        button.style.transform = 'scale(0.98)';
        button.style.transition = 'transform 0.1s ease';
        
        setTimeout(() => {
            button.style.transform = '';
        }, 100);
    }

    /**
     * Animate upload area on file drop
     */
    animateFileDrop(uploadArea) {
        if (this.prefersReducedMotion) return;

        uploadArea.style.transform = 'scale(1.02)';
        uploadArea.style.transition = 'transform 0.2s ease';
        
        setTimeout(() => {
            uploadArea.style.transform = '';
        }, 200);
    }

    /**
     * Stop all animations
     */
    stopAllAnimations() {
        // Stop particles
        this.stopAllParticles();
        
        // Clear all animation intervals
        this.activeAnimations.forEach((animation, key) => {
            if (animation.stop) {
                animation.stop();
            }
            clearInterval(animation);
            clearTimeout(animation);
        });
        
        this.activeAnimations.clear();
        
        // Remove CSS animations
        const animatedElements = document.querySelectorAll('[style*="animation"]');
        animatedElements.forEach(element => {
            element.style.animation = '';
        });
    }

    /**
     * Create custom CSS animation
     */
    createCustomAnimation(element, keyframes, options = {}) {
        if (this.prefersReducedMotion) return Promise.resolve();

        const {
            duration = CONFIG.ANIMATION.DURATION.NORMAL,
            easing = CONFIG.ANIMATION.EASING,
            iterations = 1,
            fill = 'forwards'
        } = options;

        if (element.animate) {
            // Use Web Animations API if available
            const animation = element.animate(keyframes, {
                duration,
                easing,
                iterations,
                fill
            });
            
            return animation.finished;
        } else {
            // Fallback to CSS transitions
            return new Promise((resolve) => {
                element.style.transition = `all ${duration}ms ${easing}`;
                
                // Apply final state
                const finalFrame = keyframes[keyframes.length - 1];
                Object.assign(element.style, finalFrame);
                
                setTimeout(resolve, duration);
            });
        }
    }

    /**
     * Animate element with spring physics
     */
    animateSpring(element, property, targetValue, options = {}) {
        if (this.prefersReducedMotion) {
            element.style[property] = targetValue;
            return Promise.resolve();
        }

        const {
            stiffness = 0.1,
            damping = 0.8,
            precision = 0.01
        } = options;

        return new Promise((resolve) => {
            let currentValue = parseFloat(getComputedStyle(element)[property]) || 0;
            let velocity = 0;
            
            const animate = () => {
                const force = (targetValue - currentValue) * stiffness;
                velocity += force;
                velocity *= damping;
                currentValue += velocity;
                
                element.style[property] = currentValue + (property.includes('px') ? 'px' : '');
                
                if (Math.abs(targetValue - currentValue) > precision || Math.abs(velocity) > precision) {
                    requestAnimationFrame(animate);
                } else {
                    element.style[property] = targetValue;
                    resolve();
                }
            };
            
            animate();
        });
    }
}

// Create global animation manager instance
const animationManager = new AnimationManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AnimationManager, animationManager };
}