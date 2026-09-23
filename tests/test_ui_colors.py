"""Test suite for UI color scheme validation.

Validates WCAG 2.1 AA compliance for gold/burnt copper/blue theme
on alabaster background.
"""

import pytest
import re
import os
from pathlib import Path


# Expected color values
EXPECTED_COLORS = {
    'alabaster': '#F8F4E1',
    'primary-gold': '#D4AF37',
    'accent-copper': '#B87333',
    'highlight-blue': '#1E88E5',
}

# WCAG 2.1 AA minimum contrast ratios
MIN_CONTRAST_TEXT = 4.5
MIN_CONTRAST_LARGE_TEXT = 3.0


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_relative_luminance(rgb):
    """Calculate relative luminance per WCAG 2.1."""
    def adjust(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [adjust(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def calculate_contrast_ratio(color1, color2):
    """Calculate contrast ratio between two colors."""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    lum1 = get_relative_luminance(rgb1)
    lum2 = get_relative_luminance(rgb2)

    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    return (lighter + 0.05) / (darker + 0.05)


class TestCSSVariables:
    """Test CSS variable definitions."""

    @pytest.fixture
    def css_content(self):
        """Read CSS file content."""
        css_path = Path('src/ui/styles.css')
        assert css_path.exists(), f"CSS file not found: {css_path}"
        return css_path.read_text()

    def test_primary_gold_variable_defined(self, css_content):
        """Verify --primary-gold CSS variable is defined."""
        pattern = r'--primary-gold:\s*#([A-Fa-f0-9]{6})'
        match = re.search(pattern, css_content)
        assert match, "--primary-gold CSS variable not found"
        assert match.group(1).upper() == 'D4AF37', \
            f"--primary-gold has wrong value: #{match.group(1)}"

    def test_accent_copper_variable_defined(self, css_content):
        """Verify --accent-copper CSS variable is defined."""
        pattern = r'--accent-copper:\s*#([A-Fa-f0-9]{6})'
        match = re.search(pattern, css_content)
        assert match, "--accent-copper CSS variable not found"
        assert match.group(1).upper() == 'B87333', \
            f"--accent-copper has wrong value: #{match.group(1)}"

    def test_highlight_blue_variable_defined(self, css_content):
        """Verify --highlight-blue CSS variable is defined."""
        pattern = r'--highlight-blue:\s*#([A-Fa-f0-9]{6})'
        match = re.search(pattern, css_content)
        assert match, "--highlight-blue CSS variable not found"
        assert match.group(1).upper() == '1E88E5', \
            f"--highlight-blue has wrong value: #{match.group(1)}"

    def test_alabaster_background_variable_defined(self, css_content):
        """Verify alabaster background variable is defined."""
        pattern = r'--alabaster:\s*#([A-Fa-f0-9]{6})'
        match = re.search(pattern, css_content)
        assert match, "--alabaster CSS variable not found"
        assert match.group(1).upper() == 'F8F4E1', \
            f"--alabaster has wrong value: #{match.group(1)}"


class TestAlabasterBackground:
    """Test alabaster background application."""

    @pytest.fixture
    def css_content(self):
        """Read CSS file content."""
        css_path = Path('src/ui/styles.css')
        return css_path.read_text()

    def test_alabaster_applied_to_body(self, css_content):
        """Verify alabaster background applied to body."""
        assert 'body' in css_content.lower(), "Body selector not found"
        assert '--alabaster' in css_content, "Alabaster variable not used"

    def test_alabaster_applied_to_containers(self, css_content):
        """Verify alabaster background applied to containers."""
        container_selectors = ['QWidget', 'QMainWindow', 'QFrame', 'QGroupBox']
        for selector in container_selectors:
            assert selector in css_content, f"Container {selector} not styled"
        assert '--alabaster' in css_content or '--bg-primary' in css_content, \
            "Alabaster background not applied to containers"


class TestColorContrast:
    """Test WCAG 2.1 AA color contrast compliance."""

    def test_text_on_alabaster_contrast(self):
        """Verify text colors have sufficient contrast on alabaster."""
        alabaster = '#F8F4E1'

        # Primary text (#212121)
        primary_text = '#212121'
        ratio = calculate_contrast_ratio(primary_text, alabaster)
        assert ratio >= MIN_CONTRAST_TEXT, \
            f"Primary text contrast {ratio:.2f}:1 below minimum {MIN_CONTRAST_TEXT}:1"

    def test_secondary_text_on_alabaster_contrast(self):
        """Verify secondary text has sufficient contrast on alabaster."""
        alabaster = '#F8F4E1'
        secondary_text = '#424242'
        ratio = calculate_contrast_ratio(secondary_text, alabaster)
        assert ratio >= MIN_CONTRAST_TEXT, \
            f"Secondary text contrast {ratio:.2f}:1 below minimum {MIN_CONTRAST_TEXT}:1"

    def test_text_on_gold_contrast(self):
        """Verify text on gold has sufficient contrast."""
        gold = '#D4AF37'
        text_on_gold = '#212121'
        ratio = calculate_contrast_ratio(text_on_gold, gold)
        assert ratio >= MIN_CONTRAST_TEXT, \
            f"Text on gold contrast {ratio:.2f}:1 below minimum {MIN_CONTRAST_TEXT}:1"

    def test_text_on_copper_contrast(self):
        """Verify text on copper has sufficient contrast."""
        copper = '#B87333'
        text_on_copper = '#000000'
        ratio = calculate_contrast_ratio(text_on_copper, copper)
        assert ratio >= MIN_CONTRAST_TEXT, \
            f"Text on copper contrast {ratio:.2f}:1 below minimum {MIN_CONTRAST_TEXT}:1"

    def test_text_on_blue_contrast(self):
        """Verify text on blue has sufficient contrast."""
        blue = '#1E88E5'
        text_on_blue = '#000000'
        ratio = calculate_contrast_ratio(text_on_blue, blue)
        assert ratio >= MIN_CONTRAST_TEXT, \
            f"Text on blue contrast {ratio:.2f}:1 below minimum {MIN_CONTRAST_TEXT}:1"


class TestThemeToggle:
    """Test theme toggle functionality."""

    @pytest.fixture
    def theme_js_content(self):
        """Read theme.js content."""
        theme_path = Path('src/ui/theme.js')
        assert theme_path.exists(), f"Theme file not found: {theme_path}"
        return theme_path.read_text()

    def test_theme_manager_class_exists(self, theme_js_content):
        """Verify ThemeManager class is defined."""
        assert 'class ThemeManager' in theme_js_content, \
            "ThemeManager class not found in theme.js"

    def test_toggle_theme_method_exists(self, theme_js_content):
        """Verify toggleTheme method exists."""
        assert 'toggleTheme' in theme_js_content, \
            "toggleTheme method not found in theme.js"

    def test_theme_persistence_implementation(self, theme_js_content):
        """Verify theme persistence using localStorage."""
        assert 'localStorage' in theme_js_content, \
            "localStorage not used for theme persistence"
        assert 'whirlpool_theme' in theme_js_content, \
            "Theme storage key not found"

    def test_light_dark_themes_defined(self, theme_js_content):
        """Verify both light and dark themes are defined."""
        assert "'light'" in theme_js_content or '"light"' in theme_js_content, \
            "Light theme not defined"
        assert "'dark'" in theme_js_content or '"dark"' in theme_js_content, \
            "Dark theme not defined"

    def test_contrast_validation_method(self, theme_js_content):
        """Verify contrast validation method exists."""
        assert 'validateContrast' in theme_js_content or \
               'calculateContrastRatio' in theme_js_content, \
            "Contrast validation method not found"


class TestIntegration:
    """Integration tests for UI theming."""

    def test_css_and_js_files_exist(self):
        """Verify both CSS and JS theme files exist."""
        css_path = Path('src/ui/styles.css')
        js_path = Path('src/ui/theme.js')

        assert css_path.exists(), f"CSS file not found: {css_path}"
        assert js_path.exists(), f"JS file not found: {js_path}"

    def test_css_variables_match_js_config(self):
        """Verify CSS variables match JavaScript theme config."""
        css_path = Path('src/ui/styles.css')
        js_path = Path('src/ui/theme.js')

        css_content = css_path.read_text()
        js_content = js_path.read_text()

        # Check that key variables are defined in both
        key_variables = ['--alabaster', '--primary-gold', '--accent-copper',
                        '--highlight-blue', '--text-primary', '--text-secondary']

        for var in key_variables:
            assert var in css_content, f"CSS variable {var} not found"
            assert var in js_content, f"JS variable {var} not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
