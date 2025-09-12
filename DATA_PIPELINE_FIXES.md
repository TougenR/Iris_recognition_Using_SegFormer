# Data Pipeline Fixes for Iris Segmentation

## 🚨 Problem Identified
Your confusion matrix showed "all background" predictions due to critical issues in the data pipeline.

## 🔧 Critical Fixes Applied

### 1. **Fixed Mask Class Conversion** ✅
**Problem**: Only pixels with exactly value 255 became iris class, anti-aliased pixels were lost.
```python
# BEFORE (loses anti-aliased pixels)
processed_mask[mask_np == 255] = 1

# AFTER (handles anti-aliasing)
processed_mask[mask_np > 127] = 1
```

### 2. **Fixed Boundary Transform Misalignment** ✅
**Problem**: Boundary mask got different random parameters than image/mask pair.
```python
# BEFORE (separate random transforms - WRONG)
boundary_transform = A.Compose([...])  # Different random params!

# AFTER (synchronized transforms)
transform = A.Compose([...], additional_targets={'boundary': 'mask'})
```

### 3. **Fixed Subject-Aware Split** ✅
**Problem**: Splitting by camera (C1, C2...) not actual subjects.
```python
# BEFORE (camera-based, wrong)
subject_id = int(camera_match.group(1))

# AFTER (camera + session for true subject independence)
subject_id = camera_id * 1000 + session_id
```

### 4. **Removed Unrealistic Augmentations** ✅
**Problem**: Vertical flips don't make sense for eye images.
```python
# REMOVED
A.VerticalFlip(p=0.1)  # Unrealistic for eyes
```

### 5. **Added Automatic Class Balancing** ✅
**Problem**: 94.7% background vs 5.3% iris - severe imbalance.
```python
# Solution: Calculated optimal weights
weights = torch.tensor([0.5280, 9.4296])  # [background, iris]
criterion = nn.CrossEntropyLoss(weight=weights)
```

## 🚀 How to Use the Fixes

### Step 1: Calculate Class Weights
```bash
python class_weights_util.py
```
**Output:**
- Analyzes dataset class distribution
- Calculates optimal weights (0.53 for background, 9.43 for iris)
- Saves weights to `class_weights.pt`

### Step 2: Training Automatically Uses Weights
The training pipeline now automatically:
1. Loads `class_weights.pt` if available
2. Applies weights to loss function
3. Forces model to learn iris features

### Step 3: Verify Fixes
```bash
python debug_data_pipeline.py
```
**Checks:**
- Mask value consistency
- Class distribution 
- Augmentation alignment
- Visual verification

## 📊 Expected Results

### Before Fixes:
```
Confusion Matrix:
Background: 99.5% (all predictions)
Iris: 0.5% (almost nothing)
```

### After Fixes:
```
Confusion Matrix:
Background: ~85-90% (balanced)
Iris: ~10-15% (properly detected)
```

## 🔍 Technical Details

### Class Weight Calculation
```python
# Sklearn-based balanced weights
class_weights = compute_class_weight('balanced', classes=[0,1], y=all_labels)
# Result: [0.5280, 9.4296] for [background, iris]
```

### Loss Function Integration
```python
# Automatically loaded in trainer
weights_info = torch.load('class_weights.pt')
class_weights = weights_info['weight_tensor']
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Augmentation Synchronization
```python
# All targets get same random parameters
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    # ... other transforms
], additional_targets={'boundary': 'mask'})
```

## 📋 Files Modified

1. **`src/data/dataset.py`** - Fixed mask conversion and subject splitting
2. **`src/data/transforms.py`** - Fixed boundary alignment and removed bad augmentations  
3. **`src/training/trainer.py`** - Added automatic class weight loading
4. **`src/losses/combined.py`** - Support for pre-calculated weights
5. **`class_weights_util.py`** - NEW: Automatic weight calculation
6. **`debug_data_pipeline.py`** - NEW: Debugging and verification
7. **`TRAINING_GUIDE.md`** - Updated with class balancing instructions

## 🎯 Next Steps

1. **Run class weight calculation:**
   ```bash
   python class_weights_util.py
   ```

2. **Start training with fixes:**
   ```bash
   python train.py --epochs 160 --batch-size 8 --wandb
   ```

3. **Monitor training:**
   - Confusion matrix should now show proper iris predictions
   - IoU should improve significantly
   - Training loss should decrease more effectively

4. **If still having issues:**
   ```bash
   python debug_data_pipeline.py  # Verify all fixes
   ```

The severe class imbalance was the primary cause of your "all background" predictions. These fixes should resolve the issue and give you proper iris segmentation results!
