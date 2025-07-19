# Use Python 3.11 slim image for optimal performance and size
FROM --platform=linux/amd64 python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libfontconfig1-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the main extraction script
COPY extract_outline.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set Python path and optimize for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the extraction script
CMD ["python", "extract_outline.py"]
