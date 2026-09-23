from PyQt5.QtWidgets import QApplication
from src.ui.model_card import ModelCardView


def test_readable_card_removes_metadata_badges_and_keeps_examples():
    app = QApplication.instance() or QApplication([])
    original = '''---
license: apache-2.0
tags:
- generated
---
# Example model
[![Build](https://example.com/badge.svg)](https://example.com)
<!-- internal note -->
Useful description.

```python
print("<!-- keep example -->")
```
'''
    card = ModelCardView(original)
    text = card.text.toPlainText()
    assert 'license:' not in text
    assert 'internal note' not in text
    assert 'Useful description.' in text
    assert 'print("<!-- keep example -->")' in text
    assert '\ufffc' not in text
    card.source_toggle.setChecked(True)
    assert card.text.toPlainText() == original
    card.close()


def test_horizontal_rule_and_table_are_preserved():
    app = QApplication.instance() or QApplication([])
    card = ModelCardView('# Details\n\n---\n\n| Model | Score |\n|---|---|\n| Example | 42 |')
    assert 'Details' in card.text.toPlainText()
    assert 'Example' in card.text.toPlainText()
    assert '42' in card.text.toPlainText()
    card.close()
