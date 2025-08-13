# Semiconductor Wafer Image Generation

A Python system for generating synthetic wafer cross-section images with realistic semiconductor layers and etching patterns.

## 🎯 Project Overview

This project generates synthetic wafer cross-section images for:
- **Research and development** of semiconductor analysis tools
- **Training data creation** for machine learning models
- **Educational purposes** for understanding wafer structures
- **Testing and validation** of image processing algorithms

## 📁 Project Structure

```
Semi/
├── image_generation.py          # Synthetic wafer image generator
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── wafer_training_data/        # Generated training dataset
└── .venv/                     # Virtual environment
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

### 2. Generate Wafer Images

#### **macOS/Linux:**
```bash
python3 image_generation.py
```

#### **Windows:**
```cmd
python image_generation.py
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

### Output Files
- **Images**: PNG format wafer cross-sections
- **Masks**: Binary masks for each material layer
- **Metadata**: JSON files with layer information and material colors

## 🛠️ Key Scripts

### `image_generation.py`
Generates synthetic wafer cross-section images with:
- Configurable layer parameters
- Realistic material colors
- Etching pattern generation
- Metadata export for training
- Visualization tools

## 🎯 Usage Examples

### Generate New Training Data
```python
from image_generation import WaferDataGenerator

generator = WaferDataGenerator()
generator.generate_dataset(num_samples=1000)
```

### Customize Generation Parameters
```python
from image_generation import WaferDataGenerator

generator = WaferDataGenerator(
    image_size=(512, 512),
    etching_probability=0.9,
    max_layers=8
)
generator.generate_dataset(num_samples=500)
```

### Visualize Sample Images
```python
from image_generation import WaferDataGenerator

generator = WaferDataGenerator()
generator.visualize_sample(wafer_id=0)  # Show first generated wafer
```

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

### Generation Parameters
- **Image size**: 512x512 pixels
- **Max layers**: 8 per wafer
- **Etching probability**: 90%
- **Etch types**: Rectangular and trapezoidal
- **Layer thickness**: 10-50 pixels
- **Etch depth**: 10-30 pixels

## 📊 Output Structure

### Generated Files
```
wafer_training_data/
├── images/
│   ├── wafer_00000.png
│   ├── wafer_00001.png
│   └── ...
├── masks/
│   ├── wafer_00000_mask.png
│   ├── wafer_00001_mask.png
│   └── ...
└── metadata/
    ├── wafer_00000.json
    ├── wafer_00001.json
    └── ...
```

### Metadata Format
```json
{
  "wafer_id": "wafer_00000",
  "image_size": [512, 512],
  "layers": [
    {
      "material": "silicon",
      "y_start": 341,
      "y_end": 512,
      "color": [70, 130, 180]
    }
  ],
  "etching_patterns": [
    {
      "type": "rectangular",
      "x_start": 98,
      "x_end": 183,
      "y_start": 285,
      "y_end": 307
    }
  ],
  "material_colors": {
    "silicon": {"rgb": [70, 130, 180], "name": "Steel Blue"},
    "copper": {"rgb": [139, 69, 19], "name": "Saddle Brown"}
  }
}
```

## 🎉 Success Metrics

✅ **Synthetic data generation**: 1000 realistic wafer images  
✅ **Material variety**: 7 different semiconductor materials  
✅ **Etching patterns**: Rectangular and trapezoidal shapes  
✅ **Clean interfaces**: No surface roughness  
✅ **Metadata export**: Complete layer and color information  
✅ **Visualization tools**: Sample image display  

## 🚀 Next Steps

1. **Extend materials** for more semiconductor types
2. **Add more etching patterns** (circular vias, complex shapes)
3. **Include surface roughness** as optional feature
4. **Create web interface** for interactive generation
5. **Add 3D wafer generation** capabilities

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

#### **Memory Issues:**
- Reduce `num_samples` for large datasets
- Close other applications during generation
- Use smaller image sizes if needed

#### **Import Errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

---

**Project Status**: ✅ Image Generation Complete  
**Last Updated**: December 2024  
**Generated Images**: 1000+ synthetic wafer cross-sections 