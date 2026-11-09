# Stage 1: Build the React + TypeScript frontend
FROM node:20-slim AS frontend-builder
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend + serving
FROM python:3.12-slim

# Hugging Face Spaces runs as user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    EMPATHVOICE_HOST=0.0.0.0 \
    EMPATHVOICE_PORT=7860

WORKDIR $HOME/app

# Install Python runtime dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache models into the container image for fast cold-starts
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions'); \
AutoModelForSequenceClassification.from_pretrained('SamLowe/roberta-base-go_emotions'); \
AutoTokenizer.from_pretrained('RobroKools/vad-bert'); \
AutoModelForSequenceClassification.from_pretrained('RobroKools/vad-bert')"

# Copy backend code
COPY --chown=user:user app ./app
COPY --chown=user:user run.py .

# Copy built frontend assets from builder stage
COPY --chown=user:user --from=frontend-builder /frontend/dist ./frontend/dist

# Create audio outputs directory
RUN mkdir -p audio_outputs

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
