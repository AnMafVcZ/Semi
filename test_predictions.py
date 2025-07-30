#!/usr/bin/env python3
"""
Test YOLO predictions on wafer images
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def test_predictions():
    """Test the trained YOLO model on sample images"""
    
    print("Testing YOLO predictions on wafer images...")
    
    # Load the best trained model
    model_path = "wafer_lightweight/experiment_1/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    
    # Class names (in order of class IDs)
    class_names = [
        'aluminum', 'copper', 'gold', 'rectangular_etch', 
        'si3n4', 'silicon', 'sio2', 'titanium', 'trapezoidal_etch'
    ]
    
    # Test on a few sample images
    test_images = list(Path("wafer_yolo_dataset/test/images").glob("*.jpg"))[:3]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Testing on {len(test_images)} images...")
    
    for i, img_path in enumerate(test_images):
        print(f"\n{'='*50}")
        print(f"Testing image {i+1}: {img_path.name}")
        print(f"{'='*50}")
        
        # Load and display original image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make predictions
        results = model.predict(
            source=str(img_path),
            conf=0.25,  # Confidence threshold
            save=False,
            save_txt=False,
            save_conf=True
        )
        
        # Get the first result
        result = results[0]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title(f"Original Image: {img_path.name}")
        ax1.axis('off')
        
        # Annotated image with predictions
        annotated_img = img_rgb.copy()
        
        if result.boxes is not None:
            boxes = result.boxes
            print(f"  Detected {len(boxes)} objects:")
            
            # Colors for different classes
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            
            for j, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                
                # Get class name
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                color = colors[class_id] if class_id < len(colors) else 'red'
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                
                # Add label
                label = f"{class_name}: {confidence:.3f}"
                ax2.text(x1, y1-5, label, fontsize=8, color=color, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                print(f"    {j+1}. {class_name}: {confidence:.3f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        else:
            print("  No objects detected")
        
        # Show annotated image
        ax2.imshow(annotated_img)
        ax2.set_title(f"Predictions (conf > 0.25)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        if result.boxes is not None:
            boxes = result.boxes
            class_counts = {}
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"  Summary:")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count} instances")
    
    print(f"\n{'='*50}")
    print("Testing completed!")
    print(f"{'='*50}")

def test_single_image(image_path):
    """Test a single image with detailed output"""
    
    print(f"Testing single image: {image_path}")
    
    # Load model
    model_path = "wafer_lightweight/experiment_1/weights/best.pt"
    model = YOLO(model_path)
    
    # Class names
    class_names = [
        'aluminum', 'copper', 'gold', 'rectangular_etch', 
        'si3n4', 'silicon', 'sio2', 'titanium', 'trapezoidal_etch'
    ]
    
    # Make prediction
    results = model.predict(
        source=image_path,
        conf=0.1,  # Lower threshold to see more detections
        save=True,
        save_txt=True,
        save_conf=True
    )
    
    result = results[0]
    
    print(f"\nResults for {image_path}:")
    print(f"Image size: {result.orig_shape}")
    
    if result.boxes is not None:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects:")
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            print(f"  {i+1}. {class_name} (ID: {class_id})")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            print(f"     Size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
    else:
        print("No objects detected")
    
    print(f"\nResults saved to: {result.save_dir}")

if __name__ == "__main__":
    # Test multiple images
    test_predictions()
    
    # Test a single image in detail
    test_images = list(Path("wafer_yolo_dataset/test/images").glob("*.jpg"))
    if test_images:
        test_single_image(str(test_images[0])) 