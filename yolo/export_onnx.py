#!/usr/bin/env python3
import argparse
import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export YOLO Segmentation Models to ONNX')
    parser.add_argument('--weights', type=str, default='weights/yolo11n-seg.pt',
                        help='path to model weights')
    parser.add_argument('--model-type', type=str, choices=['yolo11n', 'yolo8n', 'both'], default='yolo11n',
                        help='choose model type: yolo11n, yolo8n, or both')
    parser.add_argument('--output-dir', type=str, default='weights/onnx',
                        help='output directory for ONNX models')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for ONNX optimization')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true',
                        help='dynamic ONNX axes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device for export (cpu or cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose output during export')
    return parser.parse_args()


def export_model_to_onnx(model_path, output_dir, args):
    """Export a single model to ONNX format"""
    model_name = Path(model_path).stem
    print(f"\n{'='*60}")
    print(f"Exporting {model_name} to ONNX...")
    print(f"{'='*60}")
    
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare export parameters
        export_params = {
            'format': 'onnx',
            'imgsz': args.img_size,
            'batch': args.batch_size,
            'opset': args.opset,
            'simplify': args.simplify,
            'dynamic': args.dynamic,
            'verbose': args.verbose,
            'device': args.device
        }
        
        print(f"Export parameters: {export_params}")
        
        # Export the model
        print(f"Starting ONNX export...")
        start_time = time.time()
        
        exported_model = model.export(**export_params)
        
        export_time = time.time() - start_time
        print(f"Export completed in {export_time:.2f} seconds")
        
        # Move the exported model to the desired location if needed
        if exported_model:
            output_name = f"{model_name}.onnx"
            output_path = os.path.join(output_dir, output_name)
            
            # The exported model might be in the same directory as the original
            if os.path.exists(exported_model) and exported_model != output_path:
                import shutil
                shutil.move(exported_model, output_path)
                print(f"ONNX model moved to: {output_path}")
            elif os.path.exists(output_path):
                print(f"ONNX model saved to: {output_path}")
            else:
                # Look for the ONNX file in the weights directory
                weights_onnx = model_path.replace('.pt', '.onnx')
                if os.path.exists(weights_onnx):
                    import shutil
                    shutil.move(weights_onnx, output_path)
                    print(f"ONNX model moved to: {output_path}")
                else:
                    print(f"Warning: Could not find exported ONNX model")
                    return None
            
            return output_path
        
    except Exception as e:
        print(f"Error exporting {model_name}: {e}")
        return None


def validate_onnx_model(onnx_path, original_model_path, args):
    """Validate the exported ONNX model"""
    print(f"\nValidating ONNX model: {onnx_path}")
    
    try:
        # Load original model for comparison
        original_model = YOLO(original_model_path)
        
        # Load ONNX model
        onnx_model = YOLO(onnx_path)
        
        # Create a test image
        import numpy as np
        test_img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
        
        # Test original model
        print("Testing original PyTorch model...")
        start_time = time.time()
        original_results = original_model(test_img, verbose=False)
        original_time = time.time() - start_time
        
        # Test ONNX model
        print("Testing ONNX model...")
        start_time = time.time()
        onnx_results = onnx_model(test_img, verbose=False)
        onnx_time = time.time() - start_time
        
        # Compare results
        speedup = original_time / onnx_time if onnx_time > 0 else 0
        print(f"PyTorch model inference time: {original_time:.4f}s")
        print(f"ONNX model inference time: {onnx_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Compare number of detections
        orig_detections = sum(len(result.boxes) for result in original_results) if original_results else 0
        onnx_detections = sum(len(result.boxes) for result in onnx_results) if onnx_results else 0
        print(f"PyTorch detections: {orig_detections}")
        print(f"ONNX detections: {onnx_detections}")
        
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main():
    args = parse_args()
    
    print("YOLO Segmentation ONNX Export Tool")
    print("="*50)
    
    # Check if ONNX is available
    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
    except ImportError:
        print("ONNX not found. Installing...")
        os.system("pip install onnx")
    
    # Determine which models to export
    models_to_export = []
    
    if args.model_type == 'yolo11n':
        models_to_export.append(('yolo11n', 'weights/yolo11n-seg.pt'))
    elif args.model_type == 'yolo8n':
        models_to_export.append(('yolo8n', 'weights/yolo8n-seg.pt'))
    elif args.model_type == 'both':
        models_to_export.append(('yolo11n', 'weights/yolo11n-seg.pt'))
        models_to_export.append(('yolo8n', 'weights/yolo8n-seg.pt'))
    
    # Override with custom weights if provided
    if args.weights != 'weights/yolo11n-seg.pt':
        if args.model_type == 'both':
            print("Warning: --weights argument ignored when using --model-type both")
        else:
            models_to_export[0] = (models_to_export[0][0], args.weights)
    
    # Export models
    exported_models = []
    for model_name, model_path in models_to_export:
        if not os.path.exists(model_path):
            print(f"Error: Model weights not found: {model_path}")
            continue
        
        exported_path = export_model_to_onnx(model_path, args.output_dir, args)
        if exported_path:
            exported_models.append((model_name, model_path, exported_path))
    
    # Validate exported models
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    
    for model_name, original_path, exported_path in exported_models:
        print(f"\nValidating {model_name}...")
        validate_onnx_model(exported_path, original_path, args)
    
    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"Successfully exported {len(exported_models)} model(s)")
    for model_name, _, exported_path in exported_models:
        file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
        print(f"- {model_name}: {exported_path} ({file_size:.1f} MB)")
    
    print(f"\nONNX models saved to: {args.output_dir}")
    print("You can now use these ONNX models for faster inference!")
    print("\nTo use ONNX models with the detection script:")
    print("python detect_segmentation_onnx.py --use-onnx --model-type both --save-walkability-mask --export-csv")


if __name__ == "__main__":
    main()
