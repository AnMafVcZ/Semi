# Wafer Analysis Project

A comprehensive system for generating synthetic wafer cross-section images and training YOLO object detection models to analyze semiconductor wafer structures.

## 🎯 Project Overview

This project successfully:
- ✅ Generates synthetic wafer cross-section images with realistic semiconductor layers
- ✅ Creates YOLO-compatible datasets with proper annotations
- ✅ Trains YOLO models to detect 9 different wafer components
- ✅ Achieves 83.2% mAP50 accuracy on validation data
- ✅ Provides tools for testing and analyzing custom wafer images

## 📁 Project Structure

```
Semi/
├── image_generation.py          # Synthetic wafer image generator
├── yolo.py                      # YOLO dataset converter and trainer
├── lightweight_training.py      # Lightweight training script
├── test_predictions.py          # Test model on sample images
├── test_custom_image.py         # Test model on custom images
├── analyze_custom_image.py      # Detailed analysis with visualization
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── wafer_training_data/        # Generated training dataset
├── wafer_yolo_dataset/         # YOLO-formatted dataset
├── wafer_lightweight/          # Training results and models
└── wafer_analysis_project/     # Additional training experiments
```

## 🚀 Quick Start

### 1. Setup Environment

#### **macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### **Windows:**
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note for Windows users:**
- Use `python` instead of `python3` on Windows
- Use backslashes `\` for paths on Windows
- If you get "execution policy" errors, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 2. Generate Training Data

#### **macOS/Linux:**
```bash
python3 image_generation.py
```

#### **Windows:**
```cmd
python image_generation.py
```

### 3. Train YOLO Model

#### **macOS/Linux:**
```bash
python3 lightweight_training.py
```

#### **Windows:**
```cmd
python lightweight_training.py
```

### 4. Test the Model

#### **macOS/Linux:**
```bash
# Test on sample images
python3 test_predictions.py

# Test on custom image
python3 test_custom_image.py your_image.jpg

# Detailed analysis with visualization
python3 analyze_custom_image.py your_image.jpg
```

#### **Windows:**
```cmd
# Test on sample images
python test_predictions.py

# Test on custom image
python test_custom_image.py your_image.jpg

# Detailed analysis with visualization
python analyze_custom_image.py your_image.jpg
```

## 🎨 Generated Data Features

### Synthetic Wafer Images
- **Realistic semiconductor layers**: Silicon, SiO2, Si3N4, metals (Al, Cu, Au, Ti)
- **Etching patterns**: Rectangular and trapezoidal shapes
- **Clean interfaces**: No surface roughness for clear detection
- **90% etching probability**: Most wafers have etching patterns
- **Consistent etching**: Same height/depth within each sample
- **Surface-only etching**: Patterns only on material surfaces

### Dataset Statistics
- **Training set**: 700 images
- **Validation set**: 200 images  
- **Test set**: 100 images
- **Total**: 1000 synthetic wafer images

## 🤖 YOLO Model Performance

### Model Architecture
- **Model**: YOLOv8n (nano version)
- **Parameters**: 3M parameters
- **Training time**: ~1.75 hours on CPU
- **Inference speed**: ~20ms per image

### Detection Classes (9 total)
1. **aluminum** - Aluminum layers
2. **copper** - Copper layers
3. **gold** - Gold layers
4. **rectangular_etch** - Rectangular etching patterns
5. **si3n4** - Silicon nitride layers
6. **silicon** - Silicon substrate
7. **sio2** - Silicon dioxide layers
8. **titanium** - Titanium layers
9. **trapezoidal_etch** - Trapezoidal etching patterns

### Performance Metrics
- **mAP50**: 0.832 (83.2%)
- **mAP50-95**: 0.668 (66.8%)
- **Average confidence**: 0.578
- **High confidence detections**: >0.8 threshold

## 📊 Test Results

### Sample Test Images
The model successfully detects:
- **Material layers** with high confidence (76-99%)
- **Etching patterns** (rectangular and trapezoidal)
- **Complex wafer structures** with multiple layers
- **22 objects** in complex custom images

### Custom Image Analysis
Recent test on custom image showed:
- **22 detected objects**
- **8 material layers** (Al, Cu, Si, SiO2, Si3N4)
- **14 etching patterns** (6 rectangular, 8 trapezoidal)
- **Average confidence**: 57.8%
- **High confidence detections**: 5 objects (>80%)

## 🛠️ Key Scripts

### `image_generation.py`
Generates synthetic wafer cross-section images with:
- Configurable layer parameters
- Realistic material colors
- Etching pattern generation
- Metadata export for training

### `yolo.py`
Complete YOLO workflow including:
- Dataset conversion to YOLO format
- Model training with augmentation
- Validation and testing
- Multiple experiment support

### `lightweight_training.py`
Optimized training script with:
- Smaller model (YOLOv8n)
- Reduced epochs (50)
- CPU-friendly settings
- Faster training time

### `test_predictions.py`
Comprehensive testing with:
- Sample image testing
- Confidence analysis
- Visualization
- Statistics reporting

### `analyze_custom_image.py`
Detailed analysis with:
- Side-by-side visualization
- Confidence statistics
- Class distribution analysis
- Wafer structure breakdown

## 🎯 Usage Examples

### Generate New Training Data
```python
from image_generation import WaferDataGenerator

generator = WaferDataGenerator()
generator.generate_dataset(num_samples=1000)
```

### Train YOLO Model
```python
from yolo import WaferYOLOTrainer

trainer = WaferYOLOTrainer()
results = trainer.train_model(epochs=100)
```

### Test Custom Image

#### **macOS/Linux:**
```python
from ultralytics import YOLO

model = YOLO("wafer_lightweight/experiment_1/weights/best.pt")
results = model.predict("your_image.jpg", conf=0.25)
```

#### **Windows:**
```python
from ultralytics import YOLO

model = YOLO("wafer_lightweight\\experiment_1\\weights\\best.pt")
results = model.predict("your_image.jpg", conf=0.25)
```

## 📈 Model Files

### Trained Models
- **Best model**: `wafer_lightweight/experiment_1/weights/best.pt`
- **Last checkpoint**: `wafer_lightweight/experiment_1/weights/last.pt`

### Training Results
- **Metrics**: `wafer_lightweight/experiment_1/results.csv`
- **Plots**: `wafer_lightweight/experiment_1/results.png`
- **Confusion matrix**: `wafer_lightweight/experiment_1/confusion_matrix.png`

## 🔧 Configuration

### Material Colors
Each material has distinct RGB colors for easy identification:
- **Silicon**: Steel Blue (70, 130, 180)
- **SiO2**: Blue (100, 149, 237)
- **Si3N4**: Green (34, 139, 34)
- **Titanium**: Orange (255, 165, 0)
- **Copper**: Brown (139, 69, 19)
- **Gold**: Yellow (255, 215, 0)
- **Aluminum**: Dark Gray (105, 105, 105)

### Training Parameters
- **Model**: YOLOv8n (nano)
- **Epochs**: 50 (lightweight) / 100 (full)
- **Batch size**: 8 (lightweight) / 16 (full)
- **Image size**: 416x416 (lightweight) / 640x640 (full)
- **Learning rate**: 0.01
- **Optimizer**: AdamW

## 🎉 Success Metrics

✅ **Synthetic data generation**: 1000 realistic wafer images  
✅ **Dataset conversion**: YOLO-compatible format  
✅ **Model training**: Successful completion  
✅ **Performance**: 83.2% mAP50 accuracy  
✅ **Testing**: Works on custom images  
✅ **Visualization**: Detailed analysis tools  

## 🚀 Next Steps

1. **Deploy model** for production use
2. **Fine-tune** on real wafer images
3. **Extend classes** for more materials
4. **Optimize** for edge devices
5. **Create API** for web interface

## 🔧 Troubleshooting

### **Windows-Specific Issues:**

#### **Execution Policy Error:**
If you get "execution policy" errors when activating the virtual environment:
```cmd
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### **Python Not Found:**
If `python` command is not recognized:
1. Download Python from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Restart your command prompt

#### **Path Issues:**
- Use backslashes `\` for Windows paths
- Use forward slashes `/` for Python code paths
- Example: `wafer_lightweight\experiment_1\weights\best.pt`

#### **Visualization Issues:**
If matplotlib doesn't display plots on Windows:
```cmd
pip install tkinter
```
Or use a different backend:
```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting
```

### **General Issues:**

#### **CUDA/GPU Issues:**
- The model works on CPU (slower but functional)
- For GPU acceleration, install CUDA toolkit
- Check GPU compatibility with PyTorch

#### **Memory Issues:**
- Reduce batch size in training scripts
- Use smaller model (YOLOv8n instead of YOLOv8s)
- Close other applications during training

---

**Project Status**: ✅ Complete and Functional  
**Last Updated**: December 2024  
**Model Performance**: Excellent (83.2% mAP50) 