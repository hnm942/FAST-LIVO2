# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-dri \
    libgthread-2.0-0 \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Create necessary directories
RUN mkdir -p results weights configs

# Download default YOLO model if not present
RUN if [ ! -f "weights/yolo11n-seg.pt" ]; then \
    mkdir -p weights && \
    wget -O weights/yolo11n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-seg.pt; \
    fi

# Create a default config file if not present
RUN if [ ! -f "configs/yolo_v12_config.yaml" ]; then \
    mkdir -p configs && \
    echo "detection:" > configs/yolo_v12_config.yaml && \
    echo "  conf_threshold: 0.25" >> configs/yolo_v12_config.yaml && \
    echo "  iou_threshold: 0.45" >> configs/yolo_v12_config.yaml && \
    echo "classes: [0, 2, 3, 5, 7]" >> configs/yolo_v12_config.yaml; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$#" -eq 0 ]; then\n\
    echo "Usage: docker run -v /path/to/images:/images -v /path/to/output:/app/results yolo-segmentation --image_path /images/your_image.jpg"\n\
    echo "Example: docker run -v $(pwd)/test_images:/images -v $(pwd)/output:/app/results yolo-segmentation --image_path /images/test.jpg"\n\
    exit 1\n\
fi\n\
python main.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port (if needed for future web interface)
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["--help"]
