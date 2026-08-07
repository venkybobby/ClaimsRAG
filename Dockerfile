FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch wheel first -- otherwise sentence-transformers pulls the
# full CUDA-enabled build (~2GB of unused nvidia-* packages on a CPU box).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model weights into the image so a cold-started machine
# doesn't have to hit the Hugging Face Hub on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src/ src/
COPY api/ api/
COPY index/ index/

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
