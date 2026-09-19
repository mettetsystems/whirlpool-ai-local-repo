FROM python:3.12-slim
WORKDIR /app
COPY requirements-runtime.txt /app/
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY src/inference/runtime.py /app/runtime.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODEL_PATH=/model
EXPOSE 8000
CMD ["python", "runtime.py"]
