# YOLO v8 Object Detector

This project uses YOLO v8 to detect static obstacles such as cars, people, and motorcycles in images and videos, while filtering out dynamic obstacles.

## Project Structure

```
├── configs/
│   └── yolo_v12_config.yaml       # Model configuration
│
├── data/
│   ├── images/                    # Input images for detection
│   ├── videos/                    # Input videos for detection
│   └── labels/                    # Labels for training/validation
│
├── weights/
│   └── yolov8n.pt                # Pretrained weights
│
├── results/
│   └── outputs/                   # Detection results
│
├── utils/
│   ├── visualizer.py              # Visualization utilities
│   └── filter_utils.py            # Object filtering utilities
│
├── detect.py                      # Main detection script for images and videos
├── detect_segmentation.py         # Instance segmentation script with colored masks
├── detect_segmentation_fixed.py   # Fixed version of segmentation script
├── extract_frames.py              # Extract frames from videos
├── extract_video_frames.py        # Simple frame extraction utility
├── filter_static.py               # Static obstacle filtering
├── download_model.py              # Script to download YOLOv8 model
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download YOLOv8 weights using the provided script:
   ```
   python download_model.py
   ```

## Usage

### Object Detection

#### Image Detection

Run detection on images:
```
python detect.py --source data/images --weights weights/yolov8n.pt --config configs/yolo_v12_config.yaml
```

#### Video Detection

Run detection on videos:
```
python detect.py --source data/videos/roads.mp4 --weights weights/yolov8n.pt --save-vid
```

#### Additional Options

- `--view-img`: Display detection results in real-time
- `--save-txt`: Save detection results to text files
- `--save-vid`: Save processed video with detections
- `--conf-thres 0.5`: Set confidence threshold (0-1)
- `--classes 0 2 3`: Filter specific classes (0: person, 2: car, 3: motorcycle)

### Frame Extraction

Extract frames from videos for processing:
```
python extract_frames.py --video data/videos/roads.mp4 --output data/images/frames
```

Or use the simpler version to extract a few frames:
```
python extract_video_frames.py --video data/videos/roads.mp4 --frames 10
```

### Static Object Filtering

Filter for static objects (cars, people, motorcycles):
```
python filter_static.py --source results/outputs --output results/filtered
```

### Instance Segmentation

Run instance segmentation on images or videos using YOLOv8 segmentation models:

```
python detect_segmentation_fixed.py --source data/images --weights weights/yolo11n-seg.pt --save-vid
```

The segmentation script provides the following features:

- Instance segmentation with colored masks for each detected object
- Automatic mask resizing to match input dimensions
- Transparency control for segmentation masks with `--alpha` parameter (0-1)
- Automatic opening of output video after processing (on macOS)
- Real-time visualization during processing
- Saving of segmentation results as images or videos

#### Additional Segmentation Options

- `--alpha 0.7`: Control mask transparency (0-1, default: 0.5)
- `--save-txt`: Save segmentation contours to text files
- `--save-vid`: Save processed video with segmentation masks
- `--conf-thres 0.5`: Set confidence threshold (0-1)
- `--classes 0 2 3`: Filter specific classes (0: person, 2: car, 3: motorcycle)
- `--export-excel`: Export inference time measurements to Excel

### Inference Time Measurement and Analysis

The segmentation script includes functionality to measure and analyze inference time:

```
python detect_segmentation.py --source data/videos/sample.mp4 --weights weights/yolo11n-seg.pt --export-excel
```

This will:
- Measure inference time for each image or video frame
- Record resolution, object count, and timestamp for each inference
- Export all measurements to an Excel file in the output directory
- Generate summary statistics (mean, min, max, std) for inference times
- Create a separate summary sheet with aggregated statistics by file type

The Excel file will be named `inference_results_YYYYMMDD_HHMMSS.xlsx` with the current timestamp.

## Classes of Interest

This project focuses on detecting:
- Cars
- People
- Motorcycles

## License

[Specify your license here]
