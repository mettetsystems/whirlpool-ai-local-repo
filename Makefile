.PHONY: setup run test smoke
setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt
run:
	.venv/bin/python -m src
test:
	QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -m 'not container'
smoke:
	QT_QPA_PLATFORM=offscreen .venv/bin/python -m src --smoke-test
