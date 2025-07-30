#!/usr/bin/env python3
"""
Test YOLO model on a custom image with detailed visualization
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_custom_image(image_path):
    """Test the trained model on a custom image with detailed visualization"""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Testing model on: {image_path}")
    
    # Load the trained model
    model_path = "wafer_lightweight/experiment_1/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    
    # Class names
    class_names = [
        'aluminum', 'copper', 'gold', 'rectangular_etch', 
        'si3n4', 'silicon', 'sio2', 'titanium', 'trapezoidal_etch'
    ]
    
    # Load and display original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make predictions
    results = model.predict(
        source=image_path,
        conf=0.25,  # Confidence threshold
        save=False,
        save_txt=False,
        save_conf=True
    )
    
    result = results[0]
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS FOR: {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"Image size: {result.orig_shape}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original image
    ax1.imshow(img_rgb)
    ax1.set_title(f"Original Image: {Path(image_path).name}", fontsize=14)
    ax1.axis('off')
    
    # Annotated image with predictions
    annotated_img = img_rgb.copy()
    
    if result.boxes is not None:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects:")
        
        # Colors for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        # Statistics
        class_counts = {}
        confidence_scores = []
        
        for j, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            
            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            color = colors[class_id] if class_id < len(colors) else 'red'
            
            # Update statistics
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(confidence)
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.3f}"
            ax2.text(x1, y1-5, label, fontsize=8, color=color, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            print(f"  {j+1:2d}. {class_name:15s}: {confidence:.3f} at ({x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f})")
    else:
        print("No objects detected")
    
    # Show annotated image
    ax2.imshow(annotated_img)
    ax2.set_title(f"Predictions (conf > 0.25)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    if result.boxes is not None:
        print(f"\n{'='*60}")
        print("DETAILED STATISTICS")
        print(f"{'='*60}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:15s}: {count:2d} instances")
        
        # Confidence statistics
        if confidence_scores:
            print(f"\nConfidence Statistics:")
            print(f"  Average confidence: {np.mean(confidence_scores):.3f}")
            print(f"  Min confidence:     {np.min(confidence_scores):.3f}")
            print(f"  Max confidence:     {np.max(confidence_scores):.3f}")
            print(f"  Std confidence:     {np.std(confidence_scores):.3f}")
        
        # High confidence detections
        high_conf_detections = [score for score in confidence_scores if score > 0.8]
        print(f"\nHigh Confidence Detections (>0.8): {len(high_conf_detections)}")
        
        # Material vs Etch analysis
        materials = ['aluminum', 'copper', 'gold', 'si3n4', 'silicon', 'sio2', 'titanium']
        etches = ['rectangular_etch', 'trapezoidal_etch']
        
        material_count = sum(class_counts.get(material, 0) for material in materials)
        etch_count = sum(class_counts.get(etch, 0) for etch in etches)
        
        print(f"\nWafer Structure Analysis:")
        print(f"  Material layers: {material_count}")
        print(f"  Etching patterns: {etch_count}")
        print(f"  Total layers: {material_count + etch_count}")
        
        # Most common classes
        if class_counts:
            most_common = max(class_counts.items(), key=lambda x: x[1])
            print(f"  Most detected class: {most_common[0]} ({most_common[1]} instances)")
    
    print(f"\n{'='*60}")
    print("Analysis completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_custom_image.py <image_path>")
        print("Example: python3 test_custom_image.py my_wafer_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_custom_image(image_path) 