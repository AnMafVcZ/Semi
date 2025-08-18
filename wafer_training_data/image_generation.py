import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import os
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from pathlib import Path

# Material color mapping (RGB values) - Distinct and recognizable colors
MATERIAL_COLORS = {
    'silicon': (128, 128, 128),     # Gray (silicon substrate)
    'sio2': (0, 100, 200),          # Blue (oxide layer)
    'si3n4': (0, 150, 0),           # Green (nitride layer)
    'titanium': (255, 140, 0),      # Orange (titanium)
    'copper': (139, 69, 19),        # Brown (copper)
    'gold': (255, 215, 0),          # Yellow (gold)
    'aluminum': (169, 169, 169),    # Dark gray (aluminum)
    'empty': (255, 255, 255)        # White (empty space)
}

# Enhanced material color information with names and hex values
MATERIAL_COLOR_INFO = {
    'silicon': {
        'rgb': (128, 128, 128),
        'name': 'Gray',
        'hex': '#808080',
        'description': 'Silicon substrate - appears as gray'
    },
    'sio2': {
        'rgb': (0, 100, 200),
        'name': 'Blue',
        'hex': '#0064C8',
        'description': 'Silicon dioxide (oxide) - appears as blue'
    },
    'si3n4': {
        'rgb': (0, 150, 0),
        'name': 'Green',
        'hex': '#009600',
        'description': 'Silicon nitride - appears as green'
    },
    'titanium': {
        'rgb': (255, 140, 0),
        'name': 'Orange',
        'hex': '#FF8C00',
        'description': 'Titanium - appears as orange'
    },
    'copper': {
        'rgb': (139, 69, 19),
        'name': 'Brown',
        'hex': '#8B4513',
        'description': 'Copper - appears as brown'
    },
    'gold': {
        'rgb': (255, 215, 0),
        'name': 'Yellow',
        'hex': '#FFD700',
        'description': 'Gold - appears as yellow'
    },
    'aluminum': {
        'rgb': (169, 169, 169),
        'name': 'Dark Gray',
        'hex': '#A9A9A9',
        'description': 'Aluminum - appears as dark gray'
    },
    'empty': {
        'rgb': (255, 255, 255),
        'name': 'White',
        'hex': '#FFFFFF',
        'description': 'Empty space/etched areas - appears as white'
    }
}

# Material IDs for labeling
MATERIAL_IDS = {
    'silicon': 1,
    'sio2': 2,
    'si3n4': 3,
    'titanium': 4,
    'copper': 5,
    'gold': 6,
    'aluminum': 7,
    'empty': 0  # Background/empty space
}

@dataclass
class LayerSpec:
    """Specification for a single layer in the wafer stack"""
    material: str
    thickness_range: Tuple[int, int]  # Min, max thickness in pixels

class WaferDataGenerator:
    """Generates synthetic wafer cross-section images with metadata"""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        
    def generate_wafer_cross_section(self, layer_specs: List[LayerSpec], 
                                   silicon_thickness: int = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a synthetic wafer cross-section image
        
        Args:
            layer_specs: List of layer specifications (bottom to top, excluding silicon)
            silicon_thickness: Thickness of silicon substrate (auto-calculated if None)
            
        Returns:
            - RGB image (color-coded materials)
            - Label mask (material IDs for each pixel)
            - Metadata dictionary with layer information
        """
        
        # Initialize images
        rgb_image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)  # White background
        label_mask = np.zeros((self.height, self.width), dtype=np.uint8)  # Background = 0
        
        # Determine silicon base thickness (typically 30-50% of image height)
        if silicon_thickness is None:
            silicon_thickness = random.randint(self.height // 3, self.height // 2)
        
        current_y = self.height - silicon_thickness
        
        # Draw silicon substrate (always at the bottom)
        rgb_image[current_y:, :] = MATERIAL_COLOR_INFO['silicon']['rgb']
        label_mask[current_y:, :] = MATERIAL_IDS['silicon']
        
        layers_info = []
        layers_info.append({
            'material': 'silicon',
            'y_start': current_y,
            'y_end': self.height,
            'thickness': silicon_thickness,
            'layer_index': 0
        })
        
        # Add layers on top of silicon (from bottom to top)
        for idx, layer_spec in enumerate(layer_specs):
            thickness = random.randint(*layer_spec.thickness_range)
            
            # Don't exceed image bounds
            if current_y - thickness < 0:
                thickness = current_y
            
            if thickness <= 0:
                break
                
            layer_start = current_y - thickness
            
            # Generate layer with clean edges
            rgb_image[layer_start:current_y, :] = MATERIAL_COLOR_INFO[layer_spec.material]['rgb']
            label_mask[layer_start:current_y, :] = MATERIAL_IDS[layer_spec.material]
            
            layers_info.append({
                'material': layer_spec.material,
                'y_start': layer_start,
                'y_end': current_y,
                'thickness': thickness,
                'layer_index': idx + 1
            })
            
            current_y = layer_start
        
        # Add etching patterns (creates empty spaces)
        if random.random() > 0.1:  # 90% chance of having etched features
            rgb_image, label_mask, etch_info = self._add_etching_patterns(rgb_image, label_mask, layers_info)
        else:
            etch_info = []
        
        # Generate comprehensive metadata
        metadata = {
            'image_dimensions': {'width': self.width, 'height': self.height},
            'layers': layers_info,
            'etching_patterns': etch_info,
            'material_colors': MATERIAL_COLOR_INFO, # Updated to include info
            'material_ids': MATERIAL_IDS,
            'statistics': self._calculate_statistics(label_mask)
        }
        
        return rgb_image, label_mask, metadata
    

    
    def _add_etching_patterns(self, rgb_image: np.ndarray, label_mask: np.ndarray, 
                            layers_info: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Add etching patterns - one type per sample, multiple instances allowed"""
        
        etch_patterns = []
        num_etches = random.randint(1, 4)
        
        # Choose one etch type for this entire sample
        etch_type = random.choice(['rectangular', 'trapezoidal'])
        
        # Calculate common etch depth for all etches in this sample
        common_depth = random.randint(15, 80)
        
        for i in range(num_etches):
            if etch_type == 'rectangular':
                pattern_info = self._add_rectangular_etch(rgb_image, label_mask, common_depth)
            elif etch_type == 'trapezoidal':
                pattern_info = self._add_trapezoidal_etch(rgb_image, label_mask, common_depth)
            
            # Only add pattern if etching was successful (found material to etch)
            if pattern_info is not None:
                pattern_info['etch_id'] = i
                etch_patterns.append(pattern_info)
        
        return rgb_image, label_mask, etch_patterns
    
    def _add_rectangular_etch(self, rgb_image: np.ndarray, label_mask: np.ndarray, max_depth: int) -> Dict:
        """Add rectangular etch pattern that starts from surface and etches through materials"""
        width = random.randint(20, 120)
        
        x_start = random.randint(0, max(1, self.width - width))
        x_end = min(x_start + width, self.width)
        
        # Find the top surface (where empty meets material)
        y_start = 0
        for y in range(self.height):
            if label_mask[y, x_start] != MATERIAL_IDS['empty']:
                y_start = y
                break
        
        # Check if the entire area at the surface is actually material (not empty)
        surface_has_material = False
        for x in range(x_start, x_end):
            if label_mask[y_start, x] != MATERIAL_IDS['empty']:
                surface_has_material = True
                break
        
        if not surface_has_material:
            return None
        
        # Calculate actual etch depth (don't etch into silicon)
        actual_depth = 0
        y_end = y_start
        
        for y in range(y_start, min(y_start + max_depth, self.height)):
            # Stop if we hit silicon or if we're still in empty space
            if label_mask[y, x_start] == MATERIAL_IDS['silicon']:
                break
            if label_mask[y, x_start] != MATERIAL_IDS['empty']:
                actual_depth += 1
            y_end = y + 1
        
        # Only create etch if we actually found material to etch
        if actual_depth > 0:
            # Create etch (empty space)
            rgb_image[y_start:y_end, x_start:x_end] = MATERIAL_COLOR_INFO['empty']['rgb']
            label_mask[y_start:y_end, x_start:x_end] = MATERIAL_IDS['empty']
            
            return {
                'type': 'rectangular',
                'coordinates': {'x_start': x_start, 'y_start': y_start, 'x_end': x_end, 'y_end': y_end},
                'dimensions': {'width': x_end - x_start, 'height': y_end - y_start}
            }
        else:
            return None
    
    def _add_trapezoidal_etch(self, rgb_image: np.ndarray, label_mask: np.ndarray, max_depth: int) -> Dict:
        """Add trapezoidal etch pattern that starts from surface and etches through materials"""
        top_width = random.randint(30, 100)
        bottom_width = random.randint(20, top_width)
        
        x_center = random.randint(top_width//2, self.width - top_width//2)
        
        # Find the top surface (where empty meets material)
        y_start = 0
        for y in range(self.height):
            if label_mask[y, x_center] != MATERIAL_IDS['empty']:
                y_start = y
                break
        
        # Check if the center area at the surface is actually material (not empty)
        if label_mask[y_start, x_center] == MATERIAL_IDS['empty']:
            return None
        
        # Calculate actual etch depth (don't etch into silicon)
        actual_depth = 0
        y_end = y_start
        
        for y in range(y_start, min(y_start + max_depth, self.height)):
            # Stop if we hit silicon
            if label_mask[y, x_center] == MATERIAL_IDS['silicon']:
                break
            if label_mask[y, x_center] != MATERIAL_IDS['empty']:
                actual_depth += 1
            y_end = y + 1
        
        # Only create etch if we actually found material to etch
        if actual_depth > 0:
            # Create trapezoidal mask
            for y in range(y_start, y_end):
                progress = (y - y_start) / max(1, actual_depth)
                current_width = int(top_width + (bottom_width - top_width) * progress)
                
                x_left = max(0, x_center - current_width // 2)
                x_right = min(self.width, x_center + current_width // 2)
                
                rgb_image[y, x_left:x_right] = MATERIAL_COLOR_INFO['empty']['rgb']
                label_mask[y, x_left:x_right] = MATERIAL_IDS['empty']
            
            return {
                'type': 'trapezoidal',
                'coordinates': {'x_center': x_center, 'y_start': y_start, 'y_end': y_end},
                'dimensions': {'top_width': top_width, 'bottom_width': bottom_width, 'height': actual_depth}
            }
        else:
            return None
    
    def _add_via_etch(self, rgb_image: np.ndarray, label_mask: np.ndarray) -> Dict:
        """Add circular via etch pattern"""
        radius = random.randint(10, 40)
        
        x_center = random.randint(radius, self.width - radius)
        y_center = random.randint(radius, self.height // 2)
        
        # Create circular mask
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
        
        rgb_image[mask] = MATERIAL_COLOR_INFO['empty']['rgb']
        label_mask[mask] = MATERIAL_IDS['empty']
        
        return {
            'type': 'via',
            'coordinates': {'x_center': x_center, 'y_center': y_center},
            'dimensions': {'radius': radius}
        }
    
    def _calculate_statistics(self, label_mask: np.ndarray) -> Dict:
        """Calculate pixel statistics for each material"""
        stats = {}
        total_pixels = label_mask.size
        
        for material, mat_id in MATERIAL_IDS.items():
            pixel_count = np.count_nonzero(label_mask == mat_id)
            percentage = (pixel_count / total_pixels) * 100
            
            stats[material] = {
                'pixel_count': int(pixel_count),
                'percentage': round(percentage, 2)
            }
        
        return stats

class WaferDatasetBuilder:
    """Main class for building the wafer cross-section dataset"""
    
    def __init__(self, output_dir: str = "wafer_dataset", image_size: Tuple[int, int] = (512, 512)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generator = WaferDataGenerator(width=image_size[0], height=image_size[1])
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
    
    def generate_dataset(self, num_samples: int = 1000, 
                        layer_config: Optional[Dict] = None):
        """
        Generate complete dataset with images, labels, and metadata
        
        Args:
            num_samples: Number of samples to generate
            layer_config: Configuration for layer generation (optional)
        """
        
        if layer_config is None:
            layer_config = self._get_default_layer_config()
        
        print(f"Generating {num_samples} wafer cross-section samples...")
        print(f"Output directory: {self.output_dir}")
        
        dataset_info = {
            'total_samples': num_samples,
            'image_size': (self.generator.width, self.generator.height),
            'material_colors': MATERIAL_COLOR_INFO, # Updated to include info
            'material_ids': MATERIAL_IDS,
            'generation_config': layer_config
        }
        
        for i in range(num_samples):
            # Generate random layer stack
            layer_specs = self._create_random_layer_stack(layer_config)
            
            # Generate wafer cross-section
            rgb_img, label_mask, metadata = self.generator.generate_wafer_cross_section(layer_specs)
            
            # Create sample name
            sample_name = f"wafer_{i:05d}"
            
            # Save RGB image
            rgb_path = self.output_dir / "images" / f"{sample_name}.png"
            Image.fromarray(rgb_img).save(rgb_path)
            
            # Save label mask
            label_path = self.output_dir / "labels" / f"{sample_name}.png"
            Image.fromarray(label_mask).save(label_path)
            
            # Save metadata
            meta_path = self.output_dir / "metadata" / f"{sample_name}.json"
            metadata['sample_name'] = sample_name
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        # Save dataset info
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Generated files:")
        print(f"  - Images: {self.output_dir}/images/")
        print(f"  - Labels: {self.output_dir}/labels/")
        print(f"  - Metadata: {self.output_dir}/metadata/")
        print(f"  - Dataset info: {self.output_dir}/dataset_info.json")
    
    def _get_default_layer_config(self) -> Dict:
        """Default configuration for layer generation"""
        return {
            'max_layers': 4,
            'min_layers': 1,
            'materials': ['sio2', 'si3n4', 'titanium', 'copper', 'gold', 'aluminum'],
            'thickness_ranges': {
                'sio2': (8, 40),
                'si3n4': (10, 35),
                'titanium': (3, 15),
                'copper': (5, 25),
                'gold': (2, 10),
                'aluminum': (5, 30)
            },

        }
    
    def _create_random_layer_stack(self, config: Dict) -> List[LayerSpec]:
        """Create a random stack of layers"""
        num_layers = random.randint(config['min_layers'], config['max_layers'])
        
        # Select random materials (no repeats)
        available_materials = config['materials'].copy()
        selected_materials = random.sample(available_materials, 
                                         min(num_layers, len(available_materials)))
        
        layer_specs = []
        for material in selected_materials:
            thickness_range = config['thickness_ranges'][material]
            
            layer_specs.append(LayerSpec(material, thickness_range))
        
        return layer_specs
    
    def visualize_sample(self, sample_idx: int = 0):
        """Visualize a generated sample"""
        
        # Load sample data
        sample_name = f"wafer_{sample_idx:05d}"
        
        try:
            rgb_img = np.array(Image.open(self.output_dir / "images" / f"{sample_name}.png"))
            label_mask = np.array(Image.open(self.output_dir / "labels" / f"{sample_name}.png"))
            
            with open(self.output_dir / "metadata" / f"{sample_name}.json", 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Sample {sample_name} not found. Generate dataset first.")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB image
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('Color-coded Materials')
        axes[0, 0].axis('off')
        
        # Label mask
        axes[0, 1].imshow(label_mask, cmap='tab10', vmin=0, vmax=7)
        axes[0, 1].set_title('Label Mask')
        axes[0, 1].axis('off')
        
        # Material legend
        legend_text = "Material Colors:\n"
        for material, color_info in MATERIAL_COLOR_INFO.items():
            legend_text += f"{material} ({color_info['name']}): RGB{color_info['rgb']}\n"
        
        axes[1, 0].text(0.05, 0.95, legend_text, transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1, 0].set_title('Color Legend')
        axes[1, 0].axis('off')
        
        # Statistics
        stats_text = "Material Statistics:\n"
        for material, stats in metadata['statistics'].items():
            stats_text += f"{material}: {stats['percentage']:.1f}%\n"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1, 1].set_title('Pixel Statistics')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Sample: {sample_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print layer information
        print(f"\nLayer Stack (bottom to top):")
        for layer in metadata['layers']:
            print(f"  {layer['layer_index']}: {layer['material']} "
                  f"(thickness: {layer['thickness']} pixels)")
        
        if metadata['etching_patterns']:
            print(f"\nEtching Patterns:")
            for etch in metadata['etching_patterns']:
                print(f"  {etch['type']}: {etch['dimensions']}")

# Example usage
if __name__ == "__main__":
    # Create dataset builder
    builder = WaferDatasetBuilder(output_dir="wafer_training_data", image_size=(512, 512))
    
    # Generate dataset
    builder.generate_dataset(num_samples=1000)
    
    # Visualize a few samples
    for i in range(3):
        builder.visualize_sample(i)