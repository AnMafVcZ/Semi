import numpy as np
import cv2
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional, Set
import random
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO
import torch
from collections import defaultdict

class WaferYOLODatasetConverter:
    """Converts synthetic wafer dataset to YOLO format using metadata"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.discovered_classes = set()
        self.class_to_id = {}
        
        # Create YOLO directory structure
        self.setup_yolo_structure()
    
    def setup_yolo_structure(self):
        """Create YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def convert_dataset(self, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1):
        """Convert synthetic dataset to YOLO format using metadata"""
        
        print("Converting synthetic wafer dataset to YOLO format...")
        
        # First pass: discover all classes from metadata
        print("Discovering classes from metadata...")
        self._discover_classes()
        
        print(f"Discovered classes: {list(self.discovered_classes)}")
        
        # Get all image files
        image_files = list((self.source_dir / "images").glob("*.png"))
        print(f"Found {len(image_files)} images")
        
        # Split dataset
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_split)
        n_val = int(len(image_files) * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Process each split
        print("Processing training set...")
        train_stats = self._process_split(train_files, 'train')
        print("Processing validation set...")
        val_stats = self._process_split(val_files, 'val')
        print("Processing test set...")
        test_stats = self._process_split(test_files, 'test')
        
        # Create data.yaml file for YOLO
        self._create_data_yaml()
        
        # Print statistics
        self._print_conversion_stats(train_stats, val_stats, test_stats)
        
        print(f"\nDataset conversion complete!")
        print(f"YOLO dataset ready at: {self.output_dir}")
        
    def _discover_classes(self):
        """Discover all unique classes from metadata files"""
        
        metadata_files = list((self.source_dir / "metadata").glob("*.json"))
        
        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Add material classes from layers
            for layer in metadata.get('layers', []):
                material = layer.get('material')
                if material and material != 'empty':
                    self.discovered_classes.add(material)
            
            # Add etch classes from etching patterns
            for etch in metadata.get('etching_patterns', []):
                etch_type = etch.get('type')
                if etch_type:
                    self.discovered_classes.add(f"{etch_type}_etch")
        
        # Create class ID mapping
        sorted_classes = sorted(list(self.discovered_classes))
        self.class_to_id = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    def _process_split(self, files: List[Path], split: str) -> Dict:
        """Process files for a specific split and return statistics"""
        
        stats = defaultdict(int)
        total_annotations = 0
        
        for img_file in files:
            # Load metadata
            sample_name = img_file.stem
            meta_file = self.source_dir / "metadata" / f"{sample_name}.json"
            
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Load image to get dimensions
            image = cv2.imread(str(img_file))
            height, width = image.shape[:2]
            
            # Generate YOLO annotations from metadata
            yolo_annotations = self._generate_yolo_annotations(metadata, width, height)
            
            # Copy image to YOLO structure (convert to JPG for YOLO)
            output_img_path = self.output_dir / split / 'images' / f"{sample_name}.jpg"
            cv2.imwrite(str(output_img_path), image)
            
            # Save YOLO labels
            output_label_path = self.output_dir / split / 'labels' / f"{sample_name}.txt"
            with open(output_label_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(f"{annotation}\n")
                    # Update stats
                    class_id = int(annotation.split()[0])
                    class_name = [k for k, v in self.class_to_id.items() if v == class_id][0]
                    stats[class_name] += 1
                    total_annotations += 1
        
        stats['total_images'] = len(files)
        stats['total_annotations'] = total_annotations
        return dict(stats)
    
    def _generate_yolo_annotations(self, metadata: Dict, img_width: int, img_height: int) -> List[str]:
        """Generate YOLO format annotations from metadata"""
        
        annotations = []
        
        # 1. Generate bounding boxes for material layers
        for layer in metadata.get('layers', []):
            material = layer.get('material')
            if material == 'empty' or material not in self.class_to_id:
                continue
            
            # Extract layer boundaries from metadata
            y_start = layer.get('y_start', 0)
            y_end = layer.get('y_end', img_height)
            thickness = layer.get('thickness', y_end - y_start)
            
            # Create bounding box for the entire layer (full width)
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = 0.5  # Full width, so center is at 0.5
            center_y = (y_start + y_end) / 2 / img_height  # Normalized center Y
            bbox_width = 1.0  # Full width
            bbox_height = thickness / img_height  # Normalized height
            
            class_id = self.class_to_id[material]
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # 2. Generate bounding boxes for etch patterns
        for etch in metadata.get('etching_patterns', []):
            etch_type = etch.get('type')
            if not etch_type:
                continue
            
            etch_class = f"{etch_type}_etch"
            if etch_class not in self.class_to_id:
                continue
            
            # Extract etch coordinates from metadata
            coords = etch.get('coordinates', {})
            dims = etch.get('dimensions', {})
            
            if etch_type == 'rectangular':
                x_start = coords.get('x_start', 0)
                y_start = coords.get('y_start', 0)
                x_end = coords.get('x_end', x_start + dims.get('width', 50))
                y_end = coords.get('y_end', y_start + dims.get('height', 50))
                
                center_x = (x_start + x_end) / 2 / img_width
                center_y = (y_start + y_end) / 2 / img_height
                bbox_width = (x_end - x_start) / img_width
                bbox_height = (y_end - y_start) / img_height
                
            elif etch_type == 'trapezoidal':
                x_center = coords.get('x_center', img_width // 2)
                y_start = coords.get('y_start', 0)
                y_end = coords.get('y_end', y_start + dims.get('height', 50))
                top_width = dims.get('top_width', 50)
                
                center_x = x_center / img_width
                center_y = (y_start + y_end) / 2 / img_height
                bbox_width = top_width / img_width
                bbox_height = (y_end - y_start) / img_height
                
            elif etch_type == 'via':
                x_center = coords.get('x_center', img_width // 2)
                y_center = coords.get('y_center', img_height // 2)
                radius = dims.get('radius', 20)
                
                center_x = x_center / img_width
                center_y = y_center / img_height
                bbox_width = (2 * radius) / img_width
                bbox_height = (2 * radius) / img_height
            
            else:
                continue
            
            # Add annotation
            class_id = self.class_to_id[etch_class]
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        return annotations
    
    def _create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_to_id),
            'names': list(self.class_to_id.keys())
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"Created data.yaml with {len(self.class_to_id)} classes")
    
    def _print_conversion_stats(self, train_stats: Dict, val_stats: Dict, test_stats: Dict):
        """Print conversion statistics"""
        
        print("\n" + "="*50)
        print("DATASET CONVERSION STATISTICS")
        print("="*50)
        
        for split_name, stats in [("TRAIN", train_stats), ("VAL", val_stats), ("TEST", test_stats)]:
            print(f"\n{split_name} SET:")
            print(f"  Images: {stats.get('total_images', 0)}")
            print(f"  Total Annotations: {stats.get('total_annotations', 0)}")
            print("  Class Distribution:")
            
            for class_name, count in sorted(stats.items()):
                if class_name not in ['total_images', 'total_annotations']:
                    print(f"    {class_name}: {count}")

class WaferYOLOTrainer:
    """Handles YOLO model training for wafer analysis using Ultralytics best practices"""
    
    def __init__(self, dataset_path: str, model_size: str = 'yolov8n.pt'):
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        self.project = "wafer_analysis"
        self.name = "wafer_experiment"
        
    def train_model(self, 
                   epochs: int = 100, 
                   imgsz: int = 640, 
                   batch: int = 16,
                   lr0: float = 0.01,
                   weight_decay: float = 0.0005,
                   momentum: float = 0.937,
                   patience: int = 100,
                   save_period: int = 10,
                   optimizer: str = 'AdamW',
                   close_mosaic: int = 10,
                   resume: bool = False,
                   pretrained: bool = True,
                   **kwargs):
        """
        Train YOLO model on wafer dataset using Ultralytics training mode
        
        Args:
            epochs: Number of training epochs
            imgsz: Input image size (pixels)
            batch: Batch size (-1 for auto batch size)
            lr0: Initial learning rate
            weight_decay: Weight decay for regularization
            momentum: SGD momentum/Adam beta1
            patience: Epochs to wait for no observable improvement
            save_period: Save checkpoint every x epochs
            optimizer: Optimizer choice ['SGD', 'Adam', 'AdamW', 'RMSProp']
            close_mosaic: Disable mosaic augmentation for final epochs
            resume: Resume training from last checkpoint
            pretrained: Use pretrained model
        """
        
        print(f"Initializing {self.model_size} model...")
        
        # Initialize model - can be pretrained or from scratch
        if pretrained:
            self.model = YOLO(self.model_size)  # Load pretrained model
        else:
            # For training from scratch, use .yaml config
            model_config = self.model_size.replace('.pt', '.yaml')
            self.model = YOLO(model_config)
        
        # Data config path
        data_yaml = self.dataset_path / 'data.yaml'
        
        print("Starting YOLO training with Ultralytics framework...")
        print(f"Dataset: {data_yaml}")
        print(f"Model: {self.model_size}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Train the model using Ultralytics train mode
        results = self.model.train(
            # Data and model settings
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            
            # Optimization settings
            optimizer=optimizer,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            
            # Training settings
            patience=patience,
            save_period=save_period,
            close_mosaic=close_mosaic,
            
            # Output settings
            project=self.project,
            name=self.name,
            exist_ok=True,
            pretrained=pretrained,
            resume=resume,
            
            # Hardware settings
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,
            
            # Validation settings
            val=True,
            save=True,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            
            # Logging
            verbose=True,
            
            # Additional custom arguments
            **kwargs
        )
        
        print("Training completed!")
        print(f"Results saved to: {self.project}/{self.name}")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print(f"Last model: {results.save_dir}/weights/last.pt")
        
        return results
    
    def train_from_scratch(self, **kwargs):
        """Train model from scratch (no pretrained weights)"""
        return self.train_model(pretrained=False, **kwargs)
    
    def resume_training(self, checkpoint_path: str = None, **kwargs):
        """Resume training from checkpoint"""
        if checkpoint_path:
            self.model = YOLO(checkpoint_path)
        return self.train_model(resume=True, **kwargs)
    
    def train_with_custom_config(self, 
                                custom_yaml: str = None, 
                                hyp_yaml: str = None, 
                                **kwargs):
        """Train with custom model or hyperparameter configuration"""
        
        if custom_yaml:
            print(f"Using custom model config: {custom_yaml}")
            self.model = YOLO(custom_yaml)
        else:
            self.model = YOLO(self.model_size)
        
        # Use custom hyperparameters if provided
        train_args = kwargs
        if hyp_yaml:
            print(f"Using custom hyperparameters: {hyp_yaml}")
            # Load custom hyperparameters
            import yaml
            with open(hyp_yaml, 'r') as f:
                custom_hyp = yaml.safe_load(f)
            train_args.update(custom_hyp)
        
        return self.train_model(**train_args)
    
    def validate_model(self, 
                      model_path: str = None,
                      split: str = 'val',
                      imgsz: int = 640,
                      batch: int = 16,
                      conf: float = 0.001,
                      iou: float = 0.6,
                      save_json: bool = True,
                      save_hybrid: bool = False,
                      **kwargs):
        """
        Validate the trained model using Ultralytics validation mode
        
        Args:
            model_path: Path to model weights (uses best.pt if None)
            split: Dataset split to validate on ['val', 'test', 'train']
            imgsz: Input image size
            batch: Batch size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save_json: Save results in COCO JSON format
            save_hybrid: Save hybrid version of labels (useful for training)
        """
        
        # Load model if not already loaded or use specific weights
        if model_path:
            model = YOLO(model_path)
        elif self.model is None:
            # Try to load best model from training
            best_model_path = f"{self.project}/{self.name}/weights/best.pt"
            if Path(best_model_path).exists():
                model = YOLO(best_model_path)
            else:
                print("No model found. Train a model first or provide model_path.")
                return None
        else:
            model = self.model
        
        data_yaml = self.dataset_path / 'data.yaml'
        
        print(f"Validating model on {split} set...")
        results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            save_json=save_json,
            save_hybrid=save_hybrid,
            project=self.project,
            name=f"{self.name}_val",
            **kwargs
        )
        
        print("Validation completed!")
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def predict_wafer(self, 
                     source,
                     model_path: str = None,
                     conf: float = 0.25,
                     iou: float = 0.7,
                     imgsz: int = 640,
                     save: bool = True,
                     save_txt: bool = True,
                     save_conf: bool = True,
                     save_crop: bool = False,
                     show: bool = False,
                     stream: bool = False,
                     **kwargs):
        """
        Make predictions on wafer images using Ultralytics predict mode
        
        Args:
            source: Path to image, directory, video, URL, or PIL/OpenCV image
            model_path: Path to model weights (uses best.pt if None)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            save: Save images with detections
            save_txt: Save detection results as .txt files
            save_conf: Save confidence scores in txt files
            save_crop: Save cropped detection images
            show: Display results
            stream: Stream mode for processing multiple images
        """
        
        # Load model if not already loaded or use specific weights
        if model_path:
            model = YOLO(model_path)
        elif self.model is None:
            # Try to load best model from training
            best_model_path = f"{self.project}/{self.name}/weights/best.pt"
            if Path(best_model_path).exists():
                model = YOLO(best_model_path)
            else:
                print("No model found. Train a model first or provide model_path.")
                return None
        else:
            model = self.model
        
        print("Making predictions...")
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            show=show,
            stream=stream,
            project=self.project,
            name=f"{self.name}_predict",
            **kwargs
        )
        
        return results
    
    def analyze_wafer_layers(self, image_path: str, conf: float = 0.25):
        """Analyze wafer layers and etches"""
        results = self.predict_wafer(image_path, conf=conf, save=False)
        
        if not results:
            return None
        
        # Parse results
        detections = results[0]
        analysis = {
            'materials_detected': [],
            'etches_detected': [],
            'layer_analysis': {}
        }
        
        # Load class names
        data_yaml = self.dataset_path / 'data.yaml'
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        class_names = data_config['names']
        
        for box in detections.boxes:
            class_id = int(box.cls.item())
            class_name = class_names[class_id]
            confidence = box.conf.item()
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            detection_info = {
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            }
            
            if 'etch' in class_name:
                analysis['etches_detected'].append(detection_info)
            else:
                analysis['materials_detected'].append(detection_info)
        
        return analysis

# Example usage following Ultralytics best practices
def main():
    """Main workflow for wafer YOLO analysis using Ultralytics best practices"""
    
    # Paths
    synthetic_data_dir = "wafer_training_data"  # Your generated dataset
    yolo_dataset_dir = "wafer_yolo_dataset"
    
    # Step 1: Convert synthetic dataset to YOLO format
    print("Step 1: Converting dataset to YOLO format...")
    converter = WaferYOLODatasetConverter(synthetic_data_dir, yolo_dataset_dir)
    converter.convert_dataset()
    
    # Step 2: Initialize trainer with project organization
    print("\nStep 2: Initializing YOLO trainer...")
    trainer = WaferYOLOTrainer(
        dataset_path=yolo_dataset_dir, 
        model_size='yolov8s.pt'  # Can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    )
    trainer.project = "wafer_analysis_project"
    trainer.name = "experiment_1"
    
    # Step 3: Train with Ultralytics best practices
    print("\nStep 3: Training YOLO model...")
    
    # Option 1: Basic training
    training_results = trainer.train_model(
        epochs=100,
        imgsz=640,
        batch=16,           # or -1 for auto batch size
        lr0=0.01,          # Initial learning rate
        optimizer='AdamW', # 'SGD', 'Adam', 'AdamW', 'RMSProp'
        patience=50,       # Early stopping patience
        save_period=10,    # Save checkpoint every 10 epochs
        close_mosaic=10,   # Disable mosaic for final 10 epochs
        
        # Data augmentation settings
        hsv_h=0.015,       # HSV-Hue augmentation
        hsv_s=0.7,         # HSV-Saturation augmentation
        hsv_v=0.4,         # HSV-Value augmentation
        degrees=0.0,       # Rotation augmentation
        translate=0.1,     # Translation augmentation
        scale=0.9,         # Scale augmentation
        fliplr=0.5,        # Left-right flip probability
        mosaic=1.0,        # Mosaic augmentation probability
        mixup=0.15,        # Mixup augmentation probability
    )
    
    # Option 2: Train from scratch (no pretrained weights)
    # training_results = trainer.train_from_scratch(epochs=200, lr0=0.01)
    
    # Option 3: Resume training from checkpoint
    # training_results = trainer.resume_training("path/to/checkpoint.pt")
    
    # Step 4: Validate model performance
    print("\nStep 4: Validating model...")
    validation_results = trainer.validate_model(
        split='val',           # Validate on validation set
        conf=0.001,           # Low confidence for comprehensive evaluation
        iou=0.6,              # IoU threshold for NMS
        save_json=True,       # Save results in COCO format
    )
    
    # Also validate on test set
    test_results = trainer.validate_model(
        split='test',
        conf=0.25,
        save_json=True
    )
    
    # Step 5: Make predictions
    print("\nStep 5: Making predictions...")
    
    # Predict on single image
    results = trainer.predict_wafer(
        source="path/to/test_wafer.jpg",
        conf=0.25,            # Confidence threshold
        iou=0.7,              # IoU threshold for NMS
        save=True,            # Save annotated images
        save_txt=True,        # Save detection results as txt
        save_conf=True,       # Include confidence in txt files
        save_crop=True,       # Save cropped detections
    )
    
    # Predict on directory of images
    batch_results = trainer.predict_wafer(
        source="path/to/test_images/",
        conf=0.25,
        save=True,
        stream=True           # Stream mode for batch processing
    )
    
    # Step 6: Advanced analysis
    print("\nStep 6: Analyzing results...")
    
    # Load the best model for analysis
    best_model_path = f"{trainer.project}/{trainer.name}/weights/best.pt"
    
    # Analyze specific wafer
    sample_analysis = trainer.analyze_wafer_layers(
        "path/to/sample_wafer.jpg", 
        conf=0.25
    )
    
    if sample_analysis:
        print("\nDetailed Wafer Analysis:")
        print(f"Materials detected: {len(sample_analysis['materials_detected'])}")
        for material in sample_analysis['materials_detected']:
            print(f"  {material['class']}: {material['confidence']:.3f} confidence")
        
        print(f"Etches detected: {len(sample_analysis['etches_detected'])}")
        for etch in sample_analysis['etches_detected']:
            print(f"  {etch['class']}: {etch['confidence']:.3f} confidence")

def train_multiple_experiments():
    """Example of training multiple experiments with different configurations"""
    
    dataset_path = "wafer_yolo_dataset"
    
    # Experiment 1: Small model, high augmentation
    trainer1 = WaferYOLOTrainer(dataset_path, 'yolov8n.pt')
    trainer1.project = "wafer_experiments"
    trainer1.name = "nano_high_aug"
    
    trainer1.train_model(
        epochs=150,
        lr0=0.01,
        mixup=0.3,
        mosaic=1.0,
        hsv_s=0.8,
        hsv_v=0.5
    )
    
    # Experiment 2: Larger model, standard augmentation
    trainer2 = WaferYOLOTrainer(dataset_path, 'yolov8m.pt')
    trainer2.project = "wafer_experiments"
    trainer2.name = "medium_standard"
    
    trainer2.train_model(
        epochs=100,
        lr0=0.01,
        batch=8,  # Smaller batch for larger model
        patience=30
    )
    
    # Experiment 3: From scratch training
    trainer3 = WaferYOLOTrainer(dataset_path, 'yolov8s.yaml')
    trainer3.project = "wafer_experiments"
    trainer3.name = "scratch_training"
    
    trainer3.train_from_scratch(
        epochs=200,
        lr0=0.001,  # Lower learning rate for scratch training
        warmup_epochs=10,
        momentum=0.9
    )

if __name__ == "__main__":
    # Run main workflow
    main()
    
    # Optionally run multiple experiments
    # train_multiple_experiments()