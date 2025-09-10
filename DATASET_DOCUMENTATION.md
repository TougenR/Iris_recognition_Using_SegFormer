# UBIRIS V2 Dataset Preprocessing for Iris Segmentation

## Overview

This document describes the preprocessing pipeline implemented for the UBIRIS V2 dataset to enable iris and pupil segmentation using SegFormer. The preprocessing converts the original dataset format into a format suitable for semantic segmentation training.

## Dataset Information

### Original UBIRIS V2 Format
- **Source**: University of Beira Interior Iris Recognition Dataset Version 2
- **Total Images**: 2,250 eye images (300×400 pixels)
- **Subjects**: 260 individuals
- **Sessions**: Multiple sessions per subject
- **Original Annotations**: Binary masks where:
  - `255` = Iris region
  - `0` = Background + Pupil region

### File Structure
```
dataset/
├── images/           # Original eye images
│   ├── C1_S1_I1.png
│   ├── C1_S1_I2.png
│   └── ...
└── masks/            # Segmentation masks
    ├── OperatorA_C1_S1_I1.png
    ├── OperatorA_C1_S1_I2.png
    └── ...
```

## Preprocessing Pipeline

### 1. Dataset Class Implementation

The `UbirisDataset` class (located in `src/data/dataset.py`) handles:

#### **Image-Mask Pairing**
- Automatically matches images with their corresponding masks
- Handles the naming convention: `C{class}_S{session}_I{image}.png` ↔ `OperatorA_C{class}_S{session}_I{image}.png`

#### **Mask Preprocessing**
```python
# Original mask values: 0 (background+pupil), 255 (iris)
# Converted to: 0 (background+pupil), 1 (iris)
processed_mask = np.zeros_like(mask_np, dtype=np.uint8)
processed_mask[mask_np == 255] = 1  # Iris pixels become class 1
# Background and pupil pixels remain class 0
```

#### **Data Splits**
- **Training**: 80% (1,800 samples)
- **Validation**: 10% (225 samples)
- **Testing**: 10% (225 samples)

### 2. Transform Pipeline

#### **Image Transforms**
```python
# Training (with augmentation)
transforms.Compose([
    transforms.Resize((512, 512)),           # SegFormer standard size
    transforms.RandomHorizontalFlip(p=0.5), # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Validation/Testing (no augmentation)
transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

#### **Mask Transforms**
```python
transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Preserve label values
    transforms.RandomHorizontalFlip(p=0.5)  # Match image augmentation
])
```

### 3. Key Features

#### **SegFormer Compatibility**
- Output format matches SegFormer requirements exactly
- Proper tensor shapes and data types
- Custom collate function for efficient batching

#### **Class Imbalance Handling**
- Automatic class distribution analysis
- Suggested class weights calculation
- Background/Pupil: ~93% of pixels
- Iris: ~7% of pixels

## Code Structure

```
src/
├── __init__.py
└── data/
    ├── __init__.py
    ├── dataset.py       # Core UbirisDataset class
    └── dataloader.py    # DataLoader utilities and transforms
```

### Core Classes and Functions

#### `UbirisDataset`
```python
dataset = UbirisDataset(
    dataset_root='dataset',
    split='train',  # 'train', 'val', or 'test'
    transform=image_transform,
    mask_transform=mask_transform
)
```

#### `get_segformer_dataloaders()`
```python
dataloaders = get_segformer_dataloaders(
    dataset_root='dataset',
    batch_size=8,
    num_workers=4
)
```

## Usage Examples

### Basic Usage

```python
from src.data import get_segformer_dataloaders

# Create dataloaders
dataloaders = get_segformer_dataloaders(
    dataset_root='dataset',
    batch_size=4,
    num_workers=2
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']

# Get a batch
batch = next(iter(train_loader))
pixel_values = batch['pixel_values']  # Shape: [4, 3, 512, 512]
labels = batch['labels']              # Shape: [4, 512, 512]
```

### SegFormer Training Setup

```python
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import torch.nn as nn

# Model configuration
config = SegformerConfig(
    num_labels=2,
    num_channels=3,
    image_size=512
)

model = SegformerForSemanticSegmentation(config)

# Loss function with class weights
class_weights = torch.tensor([0.539, 6.922])  # [background/pupil, iris]
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Dataset Statistics

### Distribution Analysis
| Metric | Value |
|--------|-------|
| Total Samples | 2,250 |
| Training Samples | 1,800 (80%) |
| Validation Samples | 225 (10%) |
| Test Samples | 225 (10%) |
| Image Resolution | 512×512 (resized from 300×400) |
| Classes | 2 |
| Background/Pupil Pixels | ~93% |
| Iris Pixels | ~7% |

### Class Definitions
| Class ID | Class Name | Description | Color in Visualization |
|----------|------------|-------------|----------------------|
| 0 | Background/Pupil | Non-iris regions including background and pupil | Black |
| 1 | Iris | Iris texture region | White/Red |

### Recommended Class Weights
Based on pixel distribution analysis:
- **Background/Pupil (Class 0)**: 0.539
- **Iris (Class 1)**: 6.922

These weights help address the significant class imbalance during training.

## File Outputs

The preprocessing pipeline generates several helpful files:

### Visualization Files
- `dataset_sample_visualization.png` - Sample image with processed mask
- `example_batch_sample_1.png` - Training batch visualization
- `example_batch_sample_2.png` - Training batch visualization
- `mask_examination_*.png` - Original mask analysis

### Test Scripts
- `test_dataset.py` - Basic dataset functionality testing
- `test_dataloader.py` - DataLoader testing
- `example_usage.py` - Complete usage demonstration
- `debug_*.py` - Various debugging utilities

## Technical Notes

### Memory Considerations
- Images are resized to 512×512 for SegFormer compatibility
- Batch size should be adjusted based on GPU memory
- Use `num_workers=0` for debugging, higher values for training

### Data Augmentation
- Only horizontal flipping is applied to maintain anatomical consistency
- Color jittering applied only to images, not masks
- Augmentation is synchronized between images and masks

### Mask Processing
- Original masks contain only 0 and 255 values
- No separate pupil annotation in original dataset
- Pupil and background are treated as the same class
- Future work could implement pupil detection within the iris region

## Troubleshooting

### Common Issues

1. **"Unique mask values: tensor([0])"**
   - Solution: Fixed in preprocessing - masks are now properly converted

2. **Memory errors with large batch sizes**
   - Reduce batch size or use gradient accumulation
   - Recommended: Start with batch_size=4

3. **Slow data loading**
   - Increase `num_workers` parameter
   - Ensure SSD storage for dataset

### Validation

Run the test scripts to verify proper setup:
```bash
python test_dataset.py      # Basic dataset testing
python test_dataloader.py   # DataLoader testing  
python example_usage.py     # Complete pipeline test
```

## Future Enhancements

### Potential Improvements
1. **Three-class segmentation**: Separate pupil detection using morphological operations
2. **Advanced augmentation**: Rotation, scaling, elastic deformation
3. **Multi-scale training**: Different input resolutions
4. **Cross-validation**: K-fold splits for more robust evaluation

### Integration Possibilities
- **HuggingFace Integration**: Upload to HuggingFace Hub
- **Weights & Biases**: Experiment tracking integration
- **ONNX Export**: Model deployment optimization

## References

- **UBIRIS V2 Dataset**: [Official Paper/Website]
- **SegFormer**: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- **HuggingFace Transformers**: [SegFormer Documentation]

---

**Last Updated**: September 10, 2025  
**Version**: 1.0  
**Author**: Iris Recognition Project Team
