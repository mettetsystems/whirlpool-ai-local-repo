/**
 * Whirlpool UI Theme Manager
 * Handles theme toggle and persistence across sessions
 * Uses CSS variables for gold/burnt copper/blue on alabaster background
 */

const THEME_STORAGE_KEY = 'whirlpool_theme';
const THEME_TOGGLE_KEY = 'whirlpool_theme_toggle';

/**
 * Theme configuration with WCAG 2.1 AA compliant colors
 */
const THEME_CONFIG = {
    light: {
        '--alabaster': '#F8F4E1',
        '--primary-gold': '#D4AF37',
        '--accent-copper': '#B87333',
        '--highlight-blue': '#1E88E5',
        '--text-primary': '#212121',
        '--text-secondary': '#424242',
        '--text-on-gold': '#212121',
        '--text-on-copper': '#FFFFFF',
        '--text-on-blue': '#FFFFFF',
        '--bg-primary': '#F8F4E1',
        '--bg-secondary': '#FFFFFF',
        '--bg-elevated': '#FAF9F6',
        '--border-light': '#E0E0E0',
        '--border-medium': '#BDBDBD',
        '--border-strong': '#757575',
        '--hover-gold': '#C4A030',
        '--hover-copper': '#A8682E',
        '--hover-blue': '#1976D2',
        '--focus-ring': '#1E88E5',
        '--success': '#4CAF50',
        '--warning': '#FF9800',
        '--error': '#F44336',
        '--info': '#2196F3'
    },
    dark: {
        '--alabaster': '#1A1A1A',
        '--primary-gold': '#FFD700',
        '--accent-copper': '#CD7F32',
        '--highlight-blue': '#42A5F5',
        '--text-primary': '#FFFFFF',
        '--text-secondary': '#BDBDBD',
        '--text-on-gold': '#1A1A1A',
        '--text-on-copper': '#FFFFFF',
        '--text-on-blue': '#1A1A1A',
        '--bg-primary': '#1A1A1A',
        '--bg-secondary': '#2D2D2D',
        '--bg-elevated': '#3D3D3D',
        '--border-light': '#424242',
        '--border-medium': '#616161',
        '--border-strong': '#9E9E9E',
        '--hover-gold': '#FFEB3B',
        '--hover-copper': '#E6A055',
        '--hover-blue': '#64B5F6',
        '--focus-ring': '#42A5F5',
        '--success': '#66BB6A',
        '--warning': '#FFB74D',
        '--error': '#EF5350',
        '--info': '#42A5F5'
    }
};

/**
 * ThemeManager class for handling theme operations
 */
class ThemeManager {
    constructor() {
        this.currentTheme = 'light';
        this.isDarkMode = false;
        this.init();
    }

    /**
     * Initialize theme manager
     */
    init() {
        this.loadThemePreference();
        this.applyTheme(this.currentTheme);
        this.setupThemeToggle();
    }

    /**
     * Load saved theme preference from localStorage
     */
    loadThemePreference() {
        try {
            const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
            const savedToggle = localStorage.getItem(THEME_TOGGLE_KEY);

            if (savedTheme && ['light', 'dark'].includes(savedTheme)) {
                this.currentTheme = savedTheme;
                this.isDarkMode = savedTheme === 'dark';
            } else if (savedToggle !== null) {
                // Fallback to toggle state
                this.isDarkMode = savedToggle === 'true';
                this.currentTheme = this.isDarkMode ? 'dark' : 'light';
            } else {
                // Check system preference
                this.isDarkMode = window.matchMedia &&
                    window.matchMedia('(prefers-color-scheme: dark)').matches;
                this.currentTheme = this.isDarkMode ? 'dark' : 'light';
            }
        } catch (error) {
            console.warn('ThemeManager: Could not load theme preference:', error);
            // Default to light theme
            this.isDarkMode = false;
            this.currentTheme = 'light';
        }
    }

    /**
     * Apply theme to document
     */
    applyTheme(themeName) {
        const theme = THEME_CONFIG[themeName];
        if (!theme) {
            console.error('ThemeManager: Invalid theme name:', themeName);
            return;
        }

        const root = document.documentElement;

        // Apply all CSS variables
        Object.entries(theme).forEach(([property, value]) => {
            root.style.setProperty(property, value);
        });

        // Update body class for additional styling
        document.body.classList.remove('theme-light', 'theme-dark');
        document.body.classList.add(`theme-${themeName}`);

        this.currentTheme = themeName;
        this.isDarkMode = themeName === 'dark';

        // Persist theme preference
        this.saveThemePreference();

        // Emit theme change event
        this.emitThemeChange(themeName);
    }

    /**
     * Toggle between light and dark themes
     */
    toggleTheme() {
        const newTheme = this.isDarkMode ? 'light' : 'dark';
        this.applyTheme(newTheme);
        return newTheme;
    }

    /**
     * Save theme preference to localStorage
     */
    saveThemePreference() {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, this.currentTheme);
            localStorage.setItem(THEME_TOGGLE_KEY, String(this.isDarkMode));
        } catch (error) {
            console.warn('ThemeManager: Could not save theme preference:', error);
        }
    }

    /**
     * Setup theme toggle button
     */
    setupThemeToggle() {
        // Look for existing toggle button
        let toggleButton = document.getElementById('theme-toggle');

        if (!toggleButton) {
            // Create toggle button if not exists
            toggleButton = document.createElement('button');
            toggleButton.id = 'theme-toggle';
            toggleButton.className = 'theme-toggle-button';
            toggleButton.setAttribute('aria-label', 'Toggle theme');
            toggleButton.setAttribute('title', 'Toggle theme');

            // Add icon
            const icon = document.createElement('span');
            icon.className = 'theme-toggle-icon';
            toggleButton.appendChild(icon);

            // Add to toolbar or create one
            let toolbar = document.querySelector('.toolbar, .QToolBar, .QMenuBar');
            if (!toolbar) {
                toolbar = document.createElement('div');
                toolbar.className = 'toolbar';
                document.body.insertBefore(toolbar, document.body.firstChild);
            }

            toolbar.appendChild(toggleButton);
        }

        // Update button appearance
        this.updateToggleButton();

        // Add click handler
        toggleButton.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem(THEME_STORAGE_KEY)) {
                    this.applyTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    }

    /**
     * Update toggle button appearance
     */
    updateToggleButton() {
        const toggleButton = document.getElementById('theme-toggle');
        if (!toggleButton) return;

        const icon = toggleButton.querySelector('.theme-toggle-icon');
        if (icon) {
            icon.textContent = this.isDarkMode ? '☀️' : '🌙';
        }

        toggleButton.setAttribute('aria-checked', String(this.isDarkMode));
    }

    /**
     * Emit theme change event
     */
    emitThemeChange(themeName) {
        const event = new CustomEvent('themeChange', {
            detail: { theme: themeName, isDarkMode: this.isDarkMode }
        });
        document.dispatchEvent(event);
    }

    /**
     * Get current theme
     */
    getCurrentTheme() {
        return this.currentTheme;
    }

    /**
     * Get current theme config
     */
    getCurrentThemeConfig() {
        return THEME_CONFIG[this.currentTheme];
    }

    /**
     * Validate color contrast ratios
     * @returns {Object} Validation results
     */
    validateContrast() {
        const results = {
            valid: true,
            issues: []
        };

        // WCAG 2.1 AA requirements
        const minContrastRatio = 4.5;

        // Check text colors against background
        const checks = [
            {
                name: 'Primary text on background',
                fg: this.isDarkMode ? '#FFFFFF' : '#212121',
                bg: this.isDarkMode ? '#1A1A1A' : '#F8F4E1'
            },
            {
                name: 'Secondary text on background',
                fg: this.isDarkMode ? '#BDBDBD' : '#424242',
                bg: this.isDarkMode ? '#1A1A1A' : '#F8F4E1'
            },
            {
                name: 'Text on gold',
                fg: this.isDarkMode ? '#1A1A1A' : '#212121',
                bg: this.isDarkMode ? '#FFD700' : '#D4AF37'
            },
            {
                name: 'Text on copper',
                fg: '#FFFFFF',
                bg: this.isDarkMode ? '#CD7F32' : '#B87333'
            },
            {
                name: 'Text on blue',
                fg: this.isDarkMode ? '#1A1A1A' : '#FFFFFF',
                bg: this.isDarkMode ? '#42A5F5' : '#1E88E5'
            }
        ];

        checks.forEach(check => {
            const ratio = this.calculateContrastRatio(check.fg, check.bg);
            if (ratio < minContrastRatio) {
                results.valid = false;
                results.issues.push({
                    name: check.name,
                    ratio: ratio.toFixed(2),
                    required: minContrastRatio
                });
            }
        });

        return results;
    }

    /**
     * Calculate contrast ratio between two colors
     * @param {string} hex1 - First hex color
     * @param {string} hex2 - Second hex color
     * @returns {number} Contrast ratio
     */
    calculateContrastRatio(hex1, hex2) {
        const rgb1 = this.hexToRgb(hex1);
        const rgb2 = this.hexToRgb(hex2);

        const lum1 = this.getLuminance(rgb1);
        const lum2 = this.getLuminance(rgb2);

        const lighter = Math.max(lum1, lum2);
        const darker = Math.min(lum1, lum2);

        return (lighter + 0.05) / (darker + 0.05);
    }

    /**
     * Convert hex color to RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    /**
     * Calculate relative luminance
     */
    getLuminance(rgb) {
        const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(v => {
            v = v / 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThemeManager, THEME_CONFIG };
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.whirlpoolTheme = new ThemeManager();
        });
    } else {
        window.whirlpoolTheme = new ThemeManager();
    }
}
