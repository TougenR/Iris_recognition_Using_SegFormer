"""
Visualization utilities for iris segmentation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2


def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        image: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    
    return image


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    boundary_preds: Optional[torch.Tensor] = None,
    boundary_targets: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_samples: int = 8,
    figsize_per_sample: Tuple[int, int] = (4, 4)
) -> None:
    """
    Visualize model predictions vs ground truth
    
    Args:
        images: Input images [B, 3, H, W]
        predictions: Predicted masks [B, H, W] 
        targets: Ground truth masks [B, H, W]
        boundary_preds: Boundary predictions [B, 1, H, W] (optional)
        boundary_targets: Boundary targets [B, H, W] (optional)
        save_path: Path to save visualization
        max_samples: Maximum number of samples to visualize
        figsize_per_sample: Figure size per sample
    """
    batch_size = min(images.shape[0], max_samples)
    
    # Determine number of columns based on available data
    n_cols = 4  # Image, Target, Prediction, Overlay
    if boundary_preds is not None and boundary_targets is not None:
        n_cols = 6  # Add boundary columns
    
    fig, axes = plt.subplots(
        batch_size, n_cols, 
        figsize=(n_cols * figsize_per_sample[0], batch_size * figsize_per_sample[1])
    )
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Denormalize image
        image = denormalize_image(images[i])
        prediction = predictions[i].cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions[i]
        target = targets[i].cpu().numpy() if isinstance(targets, torch.Tensor) else targets[i]
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(target, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[prediction == 1] = [1, 0, 0]  # Red for iris prediction
        overlay[target == 1] = overlay[target == 1] * 0.7 + np.array([0, 1, 0]) * 0.3  # Green tint for GT iris
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay\n(Red=Pred, Green=GT)')
        axes[i, 3].axis('off')
        
        # Boundary visualizations if available
        if boundary_preds is not None and boundary_targets is not None:
            boundary_pred = torch.sigmoid(boundary_preds[i]).squeeze().cpu().numpy()
            boundary_target = boundary_targets[i].cpu().numpy()
            
            axes[i, 4].imshow(boundary_target, cmap='hot')
            axes[i, 4].set_title('Boundary GT')
            axes[i, 4].axis('off')
            
            axes[i, 5].imshow(boundary_pred, cmap='hot')
            axes[i, 5].set_title('Boundary Pred')
            axes[i, 5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()


def create_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_dir: Optional[str] = None
) -> None:
    """
    Create comprehensive training plots
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        save_dir: Directory to save plots
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU plot
    if 'mean_iou' in train_metrics and 'mean_iou' in val_metrics:
        axes[0, 1].plot(epochs, train_metrics['mean_iou'], label='Train mIoU', marker='o')
        axes[0, 1].plot(epochs, val_metrics['mean_iou'], label='Val mIoU', marker='s')
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Target (0.90)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean IoU')
        axes[0, 1].set_title('Mean IoU Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Dice plot
    if 'mean_dice' in train_metrics and 'mean_dice' in val_metrics:
        axes[1, 0].plot(epochs, train_metrics['mean_dice'], label='Train Dice', marker='o')
        axes[1, 0].plot(epochs, val_metrics['mean_dice'], label='Val Dice', marker='s')
        axes[1, 0].axhline(y=0.93, color='r', linestyle='--', alpha=0.7, label='Target (0.93)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mean Dice')
        axes[1, 0].set_title('Mean Dice Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Class-specific IoU
    if 'class_1_iou' in train_metrics and 'class_1_iou' in val_metrics:
        axes[1, 1].plot(epochs, train_metrics['class_1_iou'], label='Train Iris IoU', marker='o')
        axes[1, 1].plot(epochs, val_metrics['class_1_iou'], label='Val Iris IoU', marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Iris IoU')
        axes[1, 1].set_title('Iris IoU Progress')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def visualize_augmentations(
    dataset,
    sample_idx: int = 0,
    num_augmentations: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize different augmentations of the same sample
    
    Args:
        dataset: Dataset with augmentation
        sample_idx: Index of sample to augment
        num_augmentations: Number of augmented versions to show
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(3, num_augmentations, figsize=(3 * num_augmentations, 9))
    
    for i in range(num_augmentations):
        sample = dataset[sample_idx]
        
        # Denormalize image
        image = denormalize_image(sample['pixel_values'])
        mask = sample['labels'].numpy()
        boundary = sample.get('boundary', torch.zeros_like(sample['labels'])).numpy()
        
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Aug {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(boundary, cmap='hot')
        axes[2, i].set_title(f'Boundary {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Augmentation visualization saved to {save_path}")
    else:
        plt.show()


def create_model_architecture_diagram(model, save_path: Optional[str] = None):
    """
    Create a diagram showing model architecture
    
    Args:
        model: PyTorch model
        save_path: Path to save diagram
    """
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 512, 512)
        
        # Forward pass
        output = model(dummy_input, return_boundary=False)
        
        # Create visualization
        dot = make_dot(output['logits'], params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"Model architecture diagram saved to {save_path}")
        else:
            dot.view()
            
    except ImportError:
        print("torchviz not available. Install with: pip install torchviz")
    except Exception as e:
        print(f"Could not create architecture diagram: {e}")


def visualize_feature_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_name: str = "segformer.encoder.block.3",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature maps from intermediate layers
    
    Args:
        model: PyTorch model
        image: Input image [1, 3, H, W]
        layer_name: Name of layer to visualize
        save_path: Path to save visualization
    """
    model.eval()
    
    # Hook to capture feature maps
    features = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                features[name] = output.detach()
            elif isinstance(output, (list, tuple)):
                features[name] = output[0].detach() if output else None
        return hook
    
    # Register hook
    target_layer = dict(model.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook_fn(layer_name))
    
    try:
        # Forward pass
        with torch.no_grad():
            _ = model(image)
        
        # Get feature maps
        if layer_name in features:
            feature_maps = features[layer_name][0]  # Take first sample
            
            # Visualize first 16 channels
            n_channels = min(16, feature_maps.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(n_channels):
                fmap = feature_maps[i].cpu().numpy()
                im = axes[i].imshow(fmap, cmap='viridis')
                axes[i].set_title(f'Channel {i+1}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_channels, 16):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps: {layer_name}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Feature maps saved to {save_path}")
            else:
                plt.show()
        else:
            print(f"Layer {layer_name} not found or no features captured")
    
    finally:
        handle.remove()


def create_class_distribution_plot(
    dataset,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Analyze and visualize class distribution in dataset
    
    Args:
        dataset: Dataset to analyze
        save_path: Path to save plot
    
    Returns:
        Class distribution statistics
    """
    print("Analyzing class distribution...")
    
    class_counts = {}
    total_pixels = 0
    
    # Sample subset of dataset for analysis (to avoid loading everything)
    sample_size = min(100, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        labels = sample['labels']
        
        unique, counts = torch.unique(labels, return_counts=True)
        
        for class_id, count in zip(unique.tolist(), counts.tolist()):
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += count
            total_pixels += count
    
    # Calculate percentages
    class_percentages = {k: (v / total_pixels) * 100 for k, v in class_counts.items()}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    classes = list(class_percentages.keys())
    percentages = list(class_percentages.values())
    class_names = ['Background/Pupil', 'Iris'][:len(classes)]
    
    bars = ax1.bar(class_names, percentages, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Class Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(percentages, labels=class_names, autopct='%1.1f%%', 
            colors=['skyblue', 'lightcoral'], startangle=90)
    ax2.set_title('Class Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    
    # Print statistics
    print("Class Distribution Analysis:")
    for class_id, class_name in enumerate(class_names):
        if class_id in class_percentages:
            print(f"  {class_name}: {class_percentages[class_id]:.2f}%")
    
    return class_percentages


def visualize_boundary_quality(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boundary_preds: torch.Tensor,
    boundary_targets: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize boundary prediction quality
    
    Args:
        predictions: Segmentation predictions [B, H, W]
        targets: Segmentation targets [B, H, W]
        boundary_preds: Boundary predictions [B, 1, H, W]
        boundary_targets: Boundary targets [B, H, W]
        save_path: Path to save visualization
    """
    batch_size = min(predictions.shape[0], 4)
    
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        pred = predictions[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        boundary_pred = torch.sigmoid(boundary_preds[i]).squeeze().cpu().numpy()
        boundary_target = boundary_targets[i].cpu().numpy()
        
        # Segmentation target
        axes[i, 0].imshow(target, cmap='gray')
        axes[i, 0].set_title('Seg Target')
        axes[i, 0].axis('off')
        
        # Segmentation prediction
        axes[i, 1].imshow(pred, cmap='gray')
        axes[i, 1].set_title('Seg Prediction')
        axes[i, 1].axis('off')
        
        # Boundary target
        axes[i, 2].imshow(boundary_target, cmap='hot')
        axes[i, 2].set_title('Boundary Target')
        axes[i, 2].axis('off')
        
        # Boundary prediction
        axes[i, 3].imshow(boundary_pred, cmap='hot')
        axes[i, 3].set_title('Boundary Prediction')
        axes[i, 3].axis('off')
        
        # Boundary error map
        boundary_pred_binary = (boundary_pred > 0.5).astype(np.uint8)
        boundary_error = np.abs(boundary_target - boundary_pred_binary)
        
        axes[i, 4].imshow(boundary_error, cmap='Reds')
        axes[i, 4].set_title('Boundary Error')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Boundary quality visualization saved to {save_path}")
    else:
        plt.show()


def create_results_summary_figure(
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive results summary figure
    
    Args:
        metrics: Dictionary of computed metrics
        confusion_matrix: Confusion matrix
        class_names: Names of classes
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, :2])
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1
    )
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Metrics bar plot
    ax2 = fig.add_subplot(gs[0, 2:])
    key_metrics = ['pixel_accuracy', 'mean_iou', 'mean_dice', 'class_1_iou', 'class_1_dice']
    metric_values = [metrics.get(m, 0) for m in key_metrics]
    metric_labels = ['Pixel Acc', 'mIoU', 'mDice', 'Iris IoU', 'Iris Dice']
    
    bars = ax2.bar(metric_labels, metric_values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    ax2.set_ylabel('Score')
    ax2.set_title('Key Metrics')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Class-wise performance
    ax3 = fig.add_subplot(gs[1, :2])
    class_metrics = ['precision', 'recall', 'f1', 'iou', 'dice']
    class_0_values = [metrics.get(f'class_0_{m}', 0) for m in class_metrics]
    class_1_values = [metrics.get(f'class_1_{m}', 0) for m in class_metrics]
    
    x = np.arange(len(class_metrics))
    width = 0.35
    
    ax3.bar(x - width/2, class_0_values, width, label='Background/Pupil', alpha=0.8)
    ax3.bar(x + width/2, class_1_values, width, label='Iris', alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Class-wise Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.title() for m in class_metrics])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance summary text
    ax4 = fig.add_subplot(gs[1:, 2:])
    ax4.axis('off')
    
    # Create performance assessment
    miou = metrics.get('mean_iou', 0)
    dice = metrics.get('mean_dice', 0)
    
    if miou >= 0.90 and dice >= 0.93:
        assessment = "✅ EXCELLENT - Meets all targets!"
        color = 'green'
    elif miou >= 0.85 and dice >= 0.90:
        assessment = "🟡 GOOD - Close to targets"
        color = 'orange'
    else:
        assessment = "❌ NEEDS IMPROVEMENT"
        color = 'red'
    
    summary_text = f"""
PERFORMANCE SUMMARY

{assessment}

Key Results:
• Mean IoU: {miou:.3f} (Target: ≥0.90)
• Mean Dice: {dice:.3f} (Target: ≥0.93)
• Iris IoU: {metrics.get('class_1_iou', 0):.3f}
• Iris Dice: {metrics.get('class_1_dice', 0):.3f}
• Boundary F1: {metrics.get('boundary_f1', 0):.3f}

Class Performance:
• Background/Pupil: {metrics.get('class_0_iou', 0):.3f} IoU
• Iris: {metrics.get('class_1_iou', 0):.3f} IoU

Overall Assessment:
{assessment}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=11,
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
    
    plt.suptitle('Iris Segmentation Results Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Results summary figure saved to {save_path}")
    else:
        plt.show()
