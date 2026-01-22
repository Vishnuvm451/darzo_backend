# 1. Start with a fresh, empty Python system (NOT your old image)
FROM python:3.9-slim

# 2. Optimization flags (Keeps logs visible and file size small)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Create the folder inside the container
WORKDIR /app

# 4. Install system tools for OpenCV (Face Recognition)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 🛑 CRITICAL STEP: Copy your NEW code into the container 🛑
COPY . .

# 7. Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]