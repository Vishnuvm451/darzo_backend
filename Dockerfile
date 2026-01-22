<<<<<<< HEAD
FROM python:3.10-slim

# System deps for OpenCV + dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
=======
FROM vrndv/darzo-apii:latest
>>>>>>> e542d9407d58014933937a13e5430015187288da

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
