#!/usr/bin/env python3
"""
Simple main script for YOLO segmentation on single images
Usage: python main.py --image_path /path/to/image.jpg
"""

import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project utilities
from utils.visualizer import Visualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Segmentation for Images')
    parser.add_argument('--image_path', type=str, 
                        help='path to input image (use this for single image)')
    parser.add_argument('--images_dir', type=str, default='/images',
                        help='directory containing images to process (default: /images)')
    parser.add_argument('--weights', type=str, default='weights/yolo11n-seg.pt',
                        help='path to model weights')
    parser.add_argument('--config', type=str, default='configs/yolo_v12_config.yaml',
                        help='path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='output directory')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='mask transparency (0-1)')
    parser.add_argument('--save-walkability-mask', action='store_true',
                        help='save walkability mask (0=non-walkable/person areas, 1=walkable areas)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        # Return default config
        return {
            'detection': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'classes': [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
        }


def create_color_map():
    """Create a vibrant color map for better visualization"""
    return {
        0: (0, 0, 255),      # person - red
        2: (0, 255, 0),      # car - green
        3: (255, 0, 0),      # motorcycle - blue
        5: (255, 255, 0),    # bus - cyan
        7: (255, 0, 255),    # truck - magenta
        1: (0, 255, 255),    # bicycle - yellow
        9: (128, 0, 255),    # traffic light - purple
        10: (255, 128, 0),   # fire hydrant - orange
        11: (0, 255, 128),   # stop sign - teal
        12: (128, 128, 255), # parking meter - light blue
    }


def apply_segmentation_mask(image, mask, color, alpha=0.5):
    """Apply a segmentation mask to an image with proper color and transparency"""
    # Create a copy of the image
    result = image.copy()
    
    # Create a colored overlay
    overlay = result.copy()
    overlay[mask > 0] = color
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result


def generate_walkability_mask(results, img_shape, person_class_id=0):
    """Generate walkability mask where 0=non-walkable (person areas), 1=walkable areas"""
    h, w = img_shape[:2]
    walkability_mask = np.ones((h, w), dtype=np.uint8)  # Start with all walkable (1)
    
    # Process results to mark person areas as non-walkable (0)
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Process each detection
            for j, class_id in enumerate(class_ids):
                if class_id == person_class_id:  # Person class
                    if j < len(result.masks):
                        # Get the mask
                        mask = result.masks[j].data.cpu().numpy()
                        
                        # Resize mask to match image dimensions
                        mask = cv2.resize(mask[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Mark person areas as non-walkable (0)
                        walkability_mask[mask > 0] = 0
    
    return walkability_mask


def process_single_image(model, img_path, conf_thres, iou_thres, args):
    """Process a single image with the segmentation model"""
    print(f"Processing image: {img_path}")
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Error: Image file not found: {img_path}")
        return False
    
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return False
    
    # Get image dimensions
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Measure inference time
    start_time = time.time()
    try:
        results = model(img_path, conf=conf_thres, iou=iou_thres)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
    except Exception as e:
        print(f"Error during inference: {e}")
        return False
    
    # Create white background image
    output_img = np.ones_like(img) * 255  # White background
    
    # Create color map (black for detections)
    color_map = create_color_map()
    
    # Count total detections
    total_detections = 0
    
    # Process results
    for i, result in enumerate(results):
        # Get detections
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        total_detections += len(boxes)
        
        # Get segmentation masks if available
        if hasattr(result, 'masks') and result.masks is not None:
            # Process each detection
            for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                # Get class name
                if hasattr(Visualizer, 'COCO_CLASSES') and 0 <= class_id < len(Visualizer.COCO_CLASSES):
                    class_name = Visualizer.COCO_CLASSES[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                # Use black color for all detections
                color = (0, 0, 0)  # Black for all detected objects
                
                # Get mask for this detection
                if j < len(result.masks):
                    # Get the mask
                    mask = result.masks[j].data.cpu().numpy()
                    
                    # Resize mask to match image dimensions
                    mask = cv2.resize(mask[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Apply the mask - make detected areas black
                    output_img[mask > 0] = color  # Set detected pixels to black
        else:
            print("No segmentation masks found in results")
    
    print(f"Total detections: {total_detections}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save output image
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(args.output, f"{base_filename}_segmented.jpg")
    cv2.imwrite(output_path, output_img)
    print(f"Segmented image saved to: {output_path}")
    
    # Generate and save walkability mask if requested
    if args.save_walkability_mask:
        walkability_mask = generate_walkability_mask(results, img.shape, person_class_id=0)
        mask_path = os.path.join(args.output, f"{base_filename}_walkability.png")
        # Save as grayscale image (0=black/non-walkable, 255=white/walkable)
        cv2.imwrite(mask_path, walkability_mask * 255)
        print(f"Walkability mask saved to: {mask_path}")
    
    return True


def process_images_from_directory(model, images_dir, conf_thres, iou_thres, args):
    """Process all images from a directory"""
    # Supported image extensions
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Find all image files
    image_files = []
    if os.path.exists(images_dir):
        for ext in img_extensions:
            image_files.extend(Path(images_dir).glob(f'*{ext}'))
            image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    success_count = 0
    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        if process_single_image(model, str(img_path), conf_thres, iou_thres, args):
            success_count += 1
    
    print(f"\nCompleted processing {success_count}/{len(image_files)} images successfully")
    return success_count > 0


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Determine processing mode
    if args.image_path:
        # Single image mode
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            sys.exit(1)
        process_single = True
    else:
        # Directory mode
        if not os.path.exists(args.images_dir):
            print(f"Error: Images directory not found: {args.images_dir}")
            sys.exit(1)
        process_single = False
    
    # Load configuration
    config = load_config(args.config)
    
    # Set detection parameters from config or command line arguments
    conf_thres = args.conf_thres if args.conf_thres is not None else config['detection']['conf_threshold']
    iou_thres = args.iou_thres if args.iou_thres is not None else config['detection']['iou_threshold']
    
    print(f"Using confidence threshold: {conf_thres}")
    print(f"Using IoU threshold: {iou_thres}")
    
    # Load model
    print(f"Loading YOLO segmentation model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        
        # Test the model with a small dummy inference
        test_img = np.zeros((320, 320, 3), dtype=np.uint8)
        test_results = model(test_img, conf=0.5, verbose=False)
        print("Model compatibility test passed")
        
        # Check if it's a segmentation model
        if not any(task in str(model.task) for task in ['segment', 'seg']):
            print(f"Warning: The model {args.weights} may not be a segmentation model.")
            print("Make sure you're using a YOLO model with the '-seg' suffix.")
        
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model from {args.weights}: {e}")
        print("Please make sure you have the correct weights file and it's compatible with YOLO segmentation.")
        sys.exit(1)
    
    # Process images based on mode
    if process_single:
        # Single image processing
        success = process_single_image(model, args.image_path, conf_thres, iou_thres, args)
        if success:
            print("Segmentation completed successfully!")
        else:
            print("Segmentation failed!")
            sys.exit(1)
    else:
        # Directory processing
        success = process_images_from_directory(model, args.images_dir, conf_thres, iou_thres, args)
        if success:
            print("Batch segmentation completed successfully!")
        else:
            print("Batch segmentation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
