FROM python:3.10-slim

WORKDIR /app

# System deps (needed for opencv + face-recognition)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
