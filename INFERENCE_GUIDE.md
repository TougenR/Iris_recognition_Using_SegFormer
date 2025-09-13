# Iris Segmentation Inference Guide

This guide explains how to use the trained SegFormer model for iris segmentation inference.

## Overview

The inference module provides easy-to-use tools for running iris segmentation on new images using your trained model. It supports both single image and batch processing with various output options.

## Files Created

### Core Inference Module
- **`src/inference/inference.py`** - Main inference module with `IrisSegmentationInference` class
- **`src/inference/__init__.py`** - Package initialization with exported classes
- **`infer.py`** - Command line interface for easy usage
- **`example_inference.py`** - Example script with different usage patterns

### Updated Files
- **`src/data/transforms.py`** - Added `get_inference_transform()` function
- **`src/utils/visualization.py`** - Added `visualize_prediction()` function

## Quick Start

### Command Line Usage

```bash
# Single image inference
python infer.py --image path/to/image.jpg --output results/

# Batch processing
python infer.py --batch dataset/test_images/ --output batch_results/

# Display results (requires matplotlib)
python infer.py --image path/to/image.jpg --show

# Use custom checkpoint
python infer.py --image path/to/image.jpg --checkpoint custom_model.pt
```

### Python API Usage

```python
from src.inference import IrisSegmentationInference

# Load model
model = IrisSegmentationInference('outputs/segformer_iris_a100/checkpoints/best.pt')

# Run inference
results = model.predict('path/to/image.jpg')

# Access results
seg_mask = results['segmentation']['mask']  # Binary mask (0=background/pupil, 1=iris)
iris_prob = results['segmentation']['iris_probability']  # Probability map
confidence = results['segmentation']['confidence']  # Confidence scores
boundary_pred = results['boundary']['boundary_probability']  # Boundary predictions

# Create overlay visualization
overlay = model.create_overlay('path/to/image.jpg', results, iris_color=(0, 255, 0))

# Save results with overlay visualizations
model.save_prediction(results, 'output_directory/', original_image='path/to/image.jpg')
```

## API Reference

### IrisSegmentationInference Class

#### Constructor
```python
IrisSegmentationInference(
    checkpoint_path: str,
    device: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None
)
```

#### Methods

**`predict(image, return_boundary=True, confidence_threshold=0.5)`**
- Run inference on a single image
- Returns dictionary with segmentation and boundary results

**`predict_batch(images, return_boundary=True, confidence_threshold=0.5)`**
- Run inference on multiple images
- Returns list of prediction results

**`save_prediction(results, output_path, original_image=None, save_overlay=True, save_comparison=True, save_components=True, overlay_kwargs=None)`**
- Save prediction results to files
- Includes mask, probability maps, boundary predictions, and overlay visualizations

**`create_overlay(image, results, **overlay_kwargs)`**
- Create overlay visualization of results on original image
- Returns overlay image as numpy array

### Results Structure

```python
results = {
    'segmentation': {
        'mask': np.ndarray,           # Binary mask [H, W]
        'probabilities': np.ndarray,   # Class probabilities [2, H, W]
        'iris_probability': np.ndarray, # Iris probability [H, W]
        'confidence': np.ndarray       # Prediction confidence [H, W]
    },
    'boundary': {  # Optional
        'boundary_probability': np.ndarray,  # Boundary probability [H, W]
        'boundary_mask': np.ndarray          # Binary boundary mask [H, W]
    },
    'original_size': Tuple[int, int],  # (width, height)
    'input_size': Tuple[int, int]      # (width, height)
}
```

## Command Line Options

### Basic Options
- `--image IMAGE` - Input image path
- `--batch BATCH` - Input directory for batch processing
- `--output OUTPUT` - Output directory (default: 'inference_results')

### Model Options
- `--checkpoint CHECKPOINT` - Path to model checkpoint (default: 'outputs/segformer_iris_a100/checkpoints/best.pt')
- `--device DEVICE` - Device to use ('cuda' or 'cpu')

### Processing Options
- `--confidence-threshold THRESHOLD` - Confidence threshold for boundary predictions (default: 0.5)
- `--no-boundary` - Skip boundary prediction processing
- `--save-components` - Save individual prediction components (default: True)

### Visualization Options
- `--show` - Display results using matplotlib
- `--save-overlay` - Save overlay visualization on original image (default: True)
- `--save-comparison` - Save side-by-side comparison visualization (default: True)
- `--iris-color` - RGB color for iris overlay (default: "255,0,0" = red)
- `--boundary-color` - RGB color for boundary overlay (default: "0,255,255" = cyan)
- `--iris-alpha` - Transparency for iris overlay (0.0-1.0, default: 0.4)
- `--boundary-alpha` - Transparency for boundary overlay (0.0-1.0, default: 0.8)

## Output Files

### Single Image Processing
- `{output_name}_mask.png` - Binary segmentation mask
- `{output_name}_iris_prob.png` - Iris probability map
- `{output_name}_boundary.png` - Boundary prediction (if enabled)
- `{output_name}_overlay.png` - Overlay visualization on original image
- `{output_name}_comparison.png` - Side-by-side comparison (original, mask, overlay)

### Batch Processing
- `result_001_{image_name}_mask.png` - Individual results
- `batch_summary.txt` - Summary report with statistics

## Examples

### 1. Basic Single Image Inference

```bash
python infer.py --image dataset/images/C100_S1_I10.png --output single_result
```

Output:
```
🔍 Processing image: dataset/images/C100_S1_I10.png
📋 Using checkpoint: outputs/segformer_iris_a100/checkpoints/best.pt
⏳ Loading model...
✅ Inference model loaded
📊 Results:
   • Image size: (300, 400)
   • Iris coverage: 11.5%
   • Average confidence: 0.994
   • Boundary density: 4.5%
💾 Predictions saved to single_result
✅ Single image inference completed!
```

### 2. Batch Processing with Custom Checkpoint

```bash
python infer.py --batch test_images/ --checkpoint custom_model.pt --output batch_results/
```

### 3. Python API Example

```python
import sys
sys.path.append('src')
from inference import IrisSegmentationInference
import numpy as np

# Load model
model = IrisSegmentationInference('outputs/segformer_iris_a100/checkpoints/best.pt')

# Process single image
results = model.predict('dataset/images/sample.png')

# Calculate statistics
seg_mask = results['segmentation']['mask']
iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
print(f"Iris coverage: {iris_coverage:.1f}%")

# Save results
model.save_prediction(results, 'my_results/')
```

## Performance Notes

- **GPU Usage**: The model automatically uses CUDA if available
- **Batch Size**: Single image processing for memory efficiency
- **Image Sizes**: Handles arbitrary input sizes (resizes to 512x512 internally)
- **Speed**: ~100-200ms per image on GPU, ~1-2s per image on CPU

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**
   - Verify the checkpoint path exists
   - Default path: `outputs/segformer_iris_a100/checkpoints/best.pt`

2. **"CUDA out of memory"**
   - Use `--device cpu` to force CPU inference
   - Process images one at a time instead of batch

3. **"Import error"**
   - Ensure you're running from the project root directory
   - Check that `src/` is in your Python path

4. **"Matplotlib not available"**
   - Install matplotlib: `pip install matplotlib`
   - Or remove `--show` flag

### Model Loading Issues

The model uses `weights_only=False` to handle checkpoint loading. This is safe for checkpoints you trained yourself but be cautious with checkpoints from unknown sources.

## Integration Tips

### Custom Processing Pipeline

```python
from src.inference import IrisSegmentationInference
import cv2
import numpy as np

def process_iris_image(image_path):
    model = IrisSegmentationInference('path/to/checkpoint.pt')
    
    # Run inference
    results = model.predict(image_path)
    
    # Custom post-processing
    mask = results['segmentation']['mask']
    
    # Extract iris region
    image = cv2.imread(image_path)
    iris_region = image * np.expand_dims(mask, axis=2)
    
    return iris_region, results
```

### Integration with Other Tools

```python
# Save in different format
from PIL import Image
import numpy as np

results = model.predict('image.jpg')
mask = results['segmentation']['mask']

# Save as different formats
Image.fromarray((mask * 255).astype(np.uint8)).save('mask.png')
np.save('mask.npy', mask)  # For further processing
```

## Advanced Usage

### Custom Model Configuration

```python
model_config = {
    'model_name': 'nvidia/segformer-b2-finetuned-ade-512-512',
    'model_type': 'enhanced',
    'num_labels': 2,
    'add_boundary_head': True
}

model = IrisSegmentationInference(
    checkpoint_path='custom_model.pt',
    model_config=model_config
)
```

### Confidence Thresholding

```python
# Higher confidence threshold for boundary detection
results = model.predict('image.jpg', confidence_threshold=0.8)

# Apply custom confidence filtering
confidence = results['segmentation']['confidence']
high_conf_mask = results['segmentation']['mask'].copy()
high_conf_mask[confidence < 0.9] = 0  # Only keep high-confidence predictions
```

## Overlay Visualization

The inference system includes comprehensive overlay visualization capabilities that allow you to visualize segmentation results directly on the original image.

### Features

- **Iris Overlay**: Highlight iris regions with customizable colors and transparency
- **Boundary Overlay**: Show iris boundaries with adjustable thickness and colors
- **Multiple Styles**: Pre-configured styles for different use cases
- **Custom Colors**: Full RGB color customization for both iris and boundary
- **Transparency Control**: Adjustable alpha values for subtle or prominent overlays

### Usage Examples

#### Basic Overlay
```python
from src.inference import IrisSegmentationInference

model = IrisSegmentationInference('checkpoint.pt')
results = model.predict('image.jpg')

# Create basic overlay (red iris, cyan boundary)
overlay = model.create_overlay('image.jpg', results)

# Save overlay
from PIL import Image
Image.fromarray(overlay).save('overlay_result.png')
```

#### Custom Overlay Styles
```python
# Clinical style (red iris, white boundary)
overlay_clinical = model.create_overlay(
    'image.jpg', results,
    iris_color=(255, 0, 0),      # Red
    boundary_color=(255, 255, 255), # White
    iris_alpha=0.4,
    boundary_alpha=0.9,
    boundary_thickness=1
)

# Scientific publication style (green iris, orange boundary)
overlay_scientific = model.create_overlay(
    'image.jpg', results,
    iris_color=(0, 255, 0),      # Green
    boundary_color=(255, 100, 0), # Orange
    iris_alpha=0.35,
    boundary_alpha=0.8,
    boundary_thickness=2
)

# Subtle style (low alpha values)
overlay_subtle = model.create_overlay(
    'image.jpg', results,
    iris_color=(0, 100, 255),    # Light blue
    iris_alpha=0.2,              # Very transparent
    boundary_alpha=0.6,
    boundary_thickness=1
)
```

#### Command Line Overlay Options
```bash
# Green iris with yellow boundary
python infer.py --image image.jpg --iris-color "0,255,0" --boundary-color "255,255,0"

# High transparency (subtle overlay)
python infer.py --image image.jpg --iris-alpha 0.2 --boundary-alpha 0.6

# Clinical style visualization
python infer.py --image image.jpg --iris-color "255,0,0" --boundary-color "255,255,255" --iris-alpha 0.4

# Iris only (no boundary)
python infer.py --image image.jpg --no-boundary
```

### Overlay Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iris_color` | Tuple[int, int, int] | (255, 0, 0) | RGB color for iris overlay |
| `boundary_color` | Tuple[int, int, int] | (0, 255, 255) | RGB color for boundary overlay |
| `iris_alpha` | float | 0.4 | Iris overlay transparency (0.0-1.0) |
| `boundary_alpha` | float | 0.8 | Boundary overlay transparency (0.0-1.0) |
| `show_boundary` | bool | True | Whether to show boundary overlay |
| `boundary_thickness` | int | 2 | Thickness of boundary lines |

### Pre-configured Styles

The system includes several pre-configured overlay styles optimized for different use cases:

1. **Clinical** - Red iris, white boundary (medical imaging)
2. **Scientific** - Green iris, orange boundary (research publications)
3. **High Contrast** - Magenta iris, green boundary (presentations)
4. **Subtle** - Light blue iris, low transparency (documentation)
5. **Iris Only** - Purple iris, no boundary (focus on segmentation)

This inference system provides a complete solution for deploying your trained iris segmentation model in production environments.
