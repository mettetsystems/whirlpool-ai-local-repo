"""Readable model documentation with the original source available on demand."""
import re
from PyQt5.QtGui import QTextDocument
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QTextEdit


def readable_card_html(markdown):
    # Hub metadata is configuration, not documentation. Only remove a complete
    # YAML header at the beginning, so ordinary horizontal rules are preserved.
    markdown = re.sub(r'\A\ufeff?---[^\S\n]*\n.*?\n(?:---|\.\.\.)[^\S\n]*(?:\n|$)', '', markdown, count=1, flags=re.S)
    # Keep examples intact while removing invisible HTML comments and web-only
    # assets from prose. Qt handles headings, tables, lists and fenced examples.
    parts = re.split(r'(^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\2[ \t]*$)', markdown, flags=re.M | re.S)
    cleaned = []
    for index in range(0, len(parts), 3):
        prose = re.sub(r'<!--.*?-->', '', parts[index], flags=re.S)
        prose = re.sub(r'<(script|style|iframe)\b[^>]*>.*?</\1\s*>', '', prose, flags=re.I | re.S)
        cleaned.append(prose)
        if index + 1 < len(parts):
            cleaned.append(parts[index + 1])
    document = QTextDocument()
    document.setMarkdown(''.join(cleaned))
    html = document.toHtml()
    # Badge/image URLs are often relative or require a browser. Suppress their
    # broken placeholders; no external resources are fetched for this view.
    html = re.sub(r'<img\b[^>]*>', '', html, flags=re.I)
    html = re.sub(r'<a\b[^>]*>\s*</a>', '', html, flags=re.I)
    return html


class ModelCardView(QWidget):
    def __init__(self, markdown, parent=None):
        super().__init__(parent)
        self.markdown = markdown
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.source_toggle = QCheckBox('Show original model card source')
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(300)
        layout.addWidget(self.source_toggle)
        layout.addWidget(self.text)
        self.source_toggle.toggled.connect(self.render)
        self.render(False)

    def render(self, source):
        if source:
            self.text.setPlainText(self.markdown)
        else:
            self.text.setHtml(readable_card_html(self.markdown))
