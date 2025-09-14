# YOLO Segmentation Main Script

This directory contains a simple main script for running YOLO segmentation on individual images using Docker.

## Files Created

- `main.py` - Simple Python script for running segmentation on a single image
- `Dockerfile` - Docker configuration for containerizing the application
- `README_main.md` - This documentation file

## Usage

### Method 1: Using Python directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run segmentation on an image
python main.py --image_path /path/to/your/image.jpg

# With custom parameters
python main.py --image_path /path/to/your/image.jpg --conf-thres 0.3 --output results --save-walkability-mask
```

### Method 2: Using Docker

#### Build the Docker image
```bash
cd /home/nguyenhoang/Downloads/FAST-LIVO2/yolo
docker build -t yolo-segmentation .
```

#### Run segmentation with Docker
```bash
# Basic usage - mount your image directory and output directory
docker run -v /path/to/your/images:/images -v /path/to/output:/app/results yolo-segmentation --image_path /images/your_image.jpg

# Example with current directory
docker run -v $(pwd)/test_images:/images -v $(pwd)/output:/app/results yolo-segmentation --image_path /images/test.jpg

# With additional parameters
docker run -v $(pwd)/test_images:/images -v $(pwd)/output:/app/results yolo-segmentation --image_path /images/test.jpg --conf-thres 0.3 --save-walkability-mask
```

## Parameters

- `--image_path`: **Required** - Path to the input image
- `--weights`: Path to model weights (default: `weights/yolo11n-seg.pt`)
- `--config`: Path to configuration file (default: `configs/yolo_v12_config.yaml`)
- `--output`: Output directory (default: `results`)
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)
- `--device`: CUDA device or 'cpu' (default: auto-detect)
- `--alpha`: Mask transparency 0-1 (default: 0.5)
- `--save-walkability-mask`: Save walkability mask for navigation

## Output Files

The script will generate:
- `{image_name}_segmented.jpg` - Image with segmentation overlays
- `{image_name}_walkability.png` - Binary walkability mask (if requested)

## Docker Volume Mounting

When using Docker, you need to mount directories:
- Mount your image directory to `/images` in the container
- Mount your desired output directory to `/app/results` in the container

## Example Commands

```bash
# Python direct usage
python main.py --image_path ../test_images/street.jpg --output results --save-walkability-mask

# Docker usage
docker run -v $(pwd)/../test_images:/images -v $(pwd)/results:/app/results yolo-segmentation --image_path /images/street.jpg --save-walkability-mask
```

## Supported Image Formats

- JPG/JPEG
- PNG  
- BMP
- TIFF
- WEBP

## Requirements

See `requirements.txt` for Python dependencies. The Docker image handles all system dependencies automatically.
