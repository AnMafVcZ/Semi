#!/usr/bin/env python3
"""
Lightweight YOLO Training for Wafer Analysis
Optimized for laptop/desktop training with reduced resource usage
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def lightweight_training():
    """Run lightweight YOLO training optimized for local machines"""
    
    print("Starting lightweight YOLO training...")
    
    # Check if dataset exists
    if not Path("wafer_yolo_dataset").exists():
        print("Error: YOLO dataset not found. Please run the dataset conversion first.")
        print("Run: python yolo.py (this will create the dataset)")
        return
    
    # Load model
    print("Loading YOLOv8n model (smallest version)...")
    model = YOLO('yolov8n.pt')  # Use nano model instead of small
    
    # Lightweight training configuration
    print("Starting training with lightweight settings...")
    results = model.train(
        data='wafer_yolo_dataset/data.yaml',
        epochs=50,              # Reduced epochs
        imgsz=416,             # Smaller image size
        batch=8,               # Smaller batch size
        lr0=0.01,
        optimizer='AdamW',
        patience=20,           # Early stopping
        save_period=10,
        device='cpu',          # Force CPU to avoid GPU memory issues
        workers=2,             # Fewer workers
        project='wafer_lightweight',
        name='experiment_1',
        exist_ok=True,
        verbose=True,
        
        # Reduced augmentation for faster training
        hsv_h=0.0,            # Disable HSV augmentation
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,          # Disable rotation
        translate=0.0,        # Disable translation
        scale=0.0,            # Disable scaling
        shear=0.0,            # Disable shear
        perspective=0.0,      # Disable perspective
        flipud=0.0,           # Disable flip up-down
        fliplr=0.5,           # Keep horizontal flip
        mosaic=0.0,           # Disable mosaic
        mixup=0.0,            # Disable mixup
        copy_paste=0.0,       # Disable copy-paste
    )
    
    print("Lightweight training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Validate the model
    print("Validating model...")
    val_results = model.val(
        data='wafer_yolo_dataset/data.yaml',
        split='val',
        conf=0.001,
        iou=0.6,
        save_json=True
    )
    
    print("Validation completed!")
    print(f"mAP50: {val_results.box.map50:.3f}")
    print(f"mAP50-95: {val_results.box.map:.3f}")
    
    return results

def quick_test():
    """Quick test to verify the model works"""
    
    print("Running quick test...")
    
    # Load the best model
    best_model_path = "wafer_lightweight/experiment_1/weights/best.pt"
    
    if Path(best_model_path).exists():
        model = YOLO(best_model_path)
        
        # Test on a sample image
        test_images = list(Path("wafer_yolo_dataset/test/images").glob("*.jpg"))
        
        if test_images:
            sample_image = str(test_images[0])
            print(f"Testing on: {sample_image}")
            
            results = model.predict(
                source=sample_image,
                conf=0.25,
                save=True,
                save_txt=True
            )
            
            print("Quick test completed!")
            print(f"Predictions saved to: {results[0].save_dir}")
        else:
            print("No test images found")
    else:
        print("Best model not found. Please run training first.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight YOLO Training")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                       help="Training mode or test mode")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        lightweight_training()
    elif args.mode == "test":
        quick_test() 