# 📊 WandB Confusion Matrix Guide for Iris Segmentation

This guide explains how to use the new WandB confusion matrix visualization functionality in your iris segmentation project.

## 🎯 Overview

The confusion matrix visualization has been integrated into the training pipeline to automatically log confusion matrices to Weights & Biases (WandB) during training. This provides visual insights into model performance and classification errors.

## 📁 Files Added

### 1. `src/utils/wandb_confusion_matrix.py`
Main module containing confusion matrix visualization functions:

- `create_wandb_confusion_matrix()` - Creates and logs confusion matrices
- `create_wandb_classification_report()` - Generates detailed classification metrics
- `log_confusion_matrix_from_metrics()` - Logs confusion matrix from IrisSegmentationMetrics
- `create_wandb_metrics_dashboard()` - Creates comprehensive metrics dashboard

### 2. `example_wandb_confusion_matrix.py` 
Example script demonstrating all functionality with sample data.

### 3. Updated Files
- `src/training/trainer.py` - Integrated confusion matrix logging
- `src/utils/__init__.py` - Added new function exports

## 🚀 Quick Start

### Automatic Integration (Recommended)

The confusion matrix logging is now **automatically integrated** into the training pipeline. When you run:

```bash
python train.py --epochs 160 --batch-size 8 --wandb
```

Confusion matrices will be automatically logged to WandB:
- **Every 10 epochs** during training
- **At the final epoch** 
- For both **training** and **validation** data

### Manual Usage

If you want to create confusion matrices manually:

```python
from utils.wandb_confusion_matrix import create_wandb_confusion_matrix

# Your predictions and targets (numpy arrays or torch tensors)
predictions = model_predictions  # Shape: [N] or [B, H, W] 
targets = ground_truth          # Shape: [N] or [B, H, W]

# Create and log confusion matrix
fig = create_wandb_confusion_matrix(
    predictions=predictions,
    targets=targets,
    class_names=['Background/Pupil', 'Iris'],
    title="Validation Confusion Matrix - Epoch 100",
    normalize=True,           # Show percentages
    log_to_wandb=True,       # Log to WandB
    save_local=True,         # Also save locally
    step=100                 # Epoch number
)
```

## 📈 What Gets Logged to WandB

### 1. Confusion Matrix Visualization
- **Normalized matrix** showing percentages and counts
- **Color-coded heatmap** with clear labels
- **Both training and validation** matrices
- **Automatic class naming** for iris segmentation

### 2. Classification Metrics Table
- **Per-class metrics**: Precision, Recall, F1-Score, Support
- **Macro averages**: Overall performance across classes
- **Weighted averages**: Accounting for class imbalance

### 3. Enhanced Metrics Dashboard
- **Target tracking**: Progress toward mIoU ≥ 0.90 and Dice ≥ 0.93
- **Performance flags**: Boolean indicators for meeting targets
- **Comprehensive logging**: All training and validation metrics

## 🎨 Visualization Features

### Class Names
Automatically configured for iris segmentation:
- **Class 0**: "Background/Pupil" 
- **Class 1**: "Iris"

### Normalization Options
- **Normalized view**: Shows percentages for easy interpretation
- **Count overlay**: Displays actual sample counts
- **Color coding**: Clear visual distinction between classes

### Error Analysis
- **False Positives**: Background pixels classified as iris
- **False Negatives**: Iris pixels classified as background
- **Visual heatmap**: Intensity shows prediction confidence

## 📊 Understanding the Results

### Typical Iris Segmentation Patterns

**Healthy Training Signs:**
- **High diagonal values** (correct predictions)
- **Low off-diagonal values** (few errors)
- **Balanced performance** between classes

**Common Issues to Watch:**
- **High false positives**: Model over-predicting iris
- **High false negatives**: Model missing iris regions  
- **Class imbalance effects**: Background dominance

### Performance Thresholds
- **Excellent**: >95% accuracy, low off-diagonal values
- **Good**: 90-95% accuracy, minimal class confusion
- **Needs improvement**: <90% accuracy, high confusion

## 🔧 Configuration Options

### Logging Frequency
Default: Every 10 epochs + final epoch

To change frequency, modify in `trainer.py`:
```python
# Log every 5 epochs instead
if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config['training']['num_epochs']:
```

### Visual Settings
- **Figure size**: Configurable in function parameters
- **Color scheme**: Uses 'Blues' colormap by default
- **Text formatting**: Bold labels with automatic color contrast

### Save Options  
- **WandB logging**: Upload to cloud dashboard
- **Local saving**: Save PNG files to disk
- **Both options**: Can be enabled simultaneously

## 📝 Example Results Interpretation

### Good Performance Matrix:
```
                    Predicted
                Background  Iris
True Background    95.2%     4.8%
     Iris         8.1%     91.9%
```
**Interpretation**: Strong performance with minimal confusion

### Poor Performance Matrix:
```
                    Predicted  
                Background  Iris
True Background    88.1%    11.9%
     Iris          23.4%    76.6%
```
**Interpretation**: High false positive rate, needs improvement

## 🛠️ Troubleshooting

### Common Issues

**1. "No predictions available in metrics object"**
- **Cause**: Metrics object not updated with data
- **Fix**: Ensure `metrics.update()` is called during training

**2. "Could not log confusion matrix to WandB"**
- **Cause**: WandB not initialized or network issues
- **Fix**: Check WandB login and internet connection

**3. "Memory issues with large datasets"**
- **Cause**: Too many samples for visualization
- **Fix**: Sample subset of predictions for visualization

### Debugging Tips

```python
# Check metrics object has data
print(f"Predictions available: {len(metrics_obj.predictions)}")
print(f"Targets available: {len(metrics_obj.targets)}")

# Test without WandB first
create_wandb_confusion_matrix(
    predictions=preds, 
    targets=targets,
    log_to_wandb=False,  # Test locally first
    save_local=True
)
```

## 🔍 Advanced Usage

### Custom Class Names
```python
create_wandb_confusion_matrix(
    predictions=predictions,
    targets=targets, 
    class_names=['Custom_Class_0', 'Custom_Class_1'],
    # ... other parameters
)
```

### Integration with Custom Metrics
```python
from evaluation.metrics import IrisSegmentationMetrics
from utils.wandb_confusion_matrix import log_confusion_matrix_from_metrics

# Your custom metrics computation
metrics = IrisSegmentationMetrics(num_classes=2)
# ... update metrics with your data ...

# Log confusion matrix
log_confusion_matrix_from_metrics(
    metrics_obj=metrics,
    epoch=current_epoch,
    phase="validation",
    class_names=['Background/Pupil', 'Iris']
)
```

## 📚 Integration with Existing Code

The new functionality seamlessly integrates with:

- **Existing trainer**: No code changes needed in main training loop
- **Current metrics**: Works with `IrisSegmentationMetrics` class
- **WandB setup**: Uses existing WandB initialization 
- **Output structure**: Follows project's file organization

## 🎯 Next Steps

1. **Run training** with `--wandb` flag to see confusion matrices
2. **Monitor WandB dashboard** for real-time updates
3. **Analyze patterns** in confusion matrices over epochs
4. **Adjust training** based on classification errors observed

---

**🚀 Ready to visualize your model's classification performance!**

The confusion matrices will help you understand:
- Where your model makes mistakes
- How performance improves over training
- Whether class imbalance affects results
- When your model has converged

Check your WandB dashboard after training starts! 📊
