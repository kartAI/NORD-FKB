import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from object_detection_dataset import COCOObjectDetectionDataset
from torchvision import transforms
import torch
from pycocotools.coco import COCO
import imageio

def get_random_colors(n_colors):
    """Generate random colors for visualization."""
    return [(random.random(), random.random(), random.random()) for _ in range(n_colors)]

def build_category_names(ann_file):
    """Build a dictionary mapping category_id to category name from COCO annotation file."""
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    return {cat['id']: cat['name'] for cat in cats}

def ensure_numpy_hwc(image):
    """Ensure image is a numpy array in (H, W, 3) format and values in [0, 1]."""
    import torch
    if isinstance(image, torch.Tensor):
        # Handle tensor format
        if image.shape[0] == 3 and image.ndim == 3:  # (C, H, W)
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.numpy()
        # Do NOT denormalize here!
        image = np.clip(image, 0, 1)
    elif isinstance(image, np.ndarray):
        # Handle numpy array format
        if image.shape[0] == 3 and image.ndim == 3:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))
        if image.max() > 1.0:
            image = image / 255.0
        image = np.clip(image, 0, 1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape after conversion: {image.shape}")
    return image

def find_visible_box_indices(dataset, min_visibility=0.5):
    """
    Return indices of images where at least one bounding box is at least min_visibility visible.
    A box is considered visible if it's not too close to the image edges.
    """
    indices = []
    for idx in range(len(dataset)):
        _, targets = dataset[idx]
        image, _ = dataset[idx]
        if isinstance(image, torch.Tensor):
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
        
        for box in targets['boxes']:
            x1, y1, x2, y2 = box
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate how much of the box is visible (not too close to edges)
            # A box is considered visible if it's not too close to any edge
            margin = 0.1  # 10% margin from edges
            visible_width = min(x2, w * (1 - margin)) - max(x1, w * margin)
            visible_height = min(y2, h * (1 - margin)) - max(y1, h * margin)
            
            if visible_width > 0 and visible_height > 0:
                visible_area = visible_width * visible_height
                box_area = box_width * box_height
                if visible_area / box_area >= min_visibility:
                    indices.append(idx)
                    break
    return indices

def find_uncluttered_indices(dataset, max_boxes=3, min_visibility=0.5):
    """Return indices of images with at most max_boxes bounding boxes and at least one visible box."""
    visible_indices = set(find_visible_box_indices(dataset, min_visibility))
    indices = []
    for idx in range(len(dataset)):
        _, targets = dataset[idx]
        if len(targets['boxes']) <= max_boxes and idx in visible_indices:
            indices.append(idx)
    return indices

def visualize_image(image, boxes, labels, category_names=None, colors=None):
    """Visualize a single image with its bounding boxes and class names, with transparent fill."""
    image = ensure_numpy_hwc(image)
    
    # Debug check
    if image.min() < 0 or image.max() > 1:
        print(f"Warning: Image values out of range [0,1]: min={image.min()}, max={image.max()}")
        image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    if colors is None:
        unique_labels = list(set([l.item() if hasattr(l, 'item') else l for l in labels]))
        colors = get_random_colors(len(unique_labels))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        label_to_color = {label: colors[label % len(colors)] for label in set(labels)}
    
    for box, label in zip(boxes, labels):
        label_val = label.item() if hasattr(label, 'item') else label
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        color = label_to_color[label_val]
        # Create rectangle patch with transparent fill
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, 
                        edgecolor=color,
                        facecolor=color + (0.15,))  # Same color, very transparent fill
        ax.add_patch(rect)
        # Add label if category names are provided
        if category_names is not None:
            label_text = category_names.get(label_val, f'Class {label_val}')
            plt.text(x1, y1 + 15, label_text, 
                    color='white',
                    fontsize=12,
                    weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.axis('off')
    return plt.gcf()

def visualize_batch(dataset, num_samples=4, category_names=None, min_visibility=0.5):
    """Visualize a batch of images from the dataset, with debug info for each image."""
    # Get random indices from visible boxes
    visible_indices = find_visible_box_indices(dataset, min_visibility)
    if len(visible_indices) < num_samples:
        print(f"Warning: Only found {len(visible_indices)} images with visible boxes. Using all available.")
        indices = visible_indices
    else:
        indices = random.sample(visible_indices, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # Generate colors for classes
    colors = get_random_colors(len(set([label for idx in indices 
                                      for label in dataset[idx][1]['labels']])))
    
    for idx, ax_idx in enumerate(indices):
        image, targets = dataset[ax_idx]
        # Save raw image for debugging
        raw_img = image.clone().cpu().numpy() if hasattr(image, 'clone') else np.copy(image)
        # Print debug info
        print(f"[DEBUG] Index: {ax_idx}, Raw shape: {raw_img.shape}, dtype: {raw_img.dtype}, min: {raw_img.min()}, max: {raw_img.max()}, mean: {raw_img.mean()}")
        # Save raw image (try to save as uint8 RGB if possible)
        try:
            if isinstance(raw_img, np.ndarray) and raw_img.ndim == 3:
                if raw_img.shape[0] == 3:  # (C, H, W)
                    img_to_save = np.transpose(raw_img, (1, 2, 0))
                else:
                    img_to_save = raw_img
                if img_to_save.max() <= 1.0:
                    img_to_save = (img_to_save * 255).astype(np.uint8)
                else:
                    img_to_save = img_to_save.astype(np.uint8)
                imageio.imwrite(f'debug_raw_{ax_idx}.png', img_to_save)
        except Exception as e:
            print(f"[DEBUG] Could not save raw image for index {ax_idx}: {e}")
        
        image = ensure_numpy_hwc(image)
        # Debug check
        print(f"[DEBUG] Post-conversion Index: {ax_idx}, shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}, mean: {image.mean()}")
        if image.min() < 0 or image.max() > 1:
            print(f"Warning: Image values out of range [0,1]: min={image.min()}, max={image.max()}")
            image = np.clip(image, 0, 1)
        
        axes[idx].imshow(image)
        
        # Draw bounding boxes
        for box, label in zip(targets['boxes'], targets['labels']):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            color = colors[label % len(colors)]
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2,
                           edgecolor=color,
                           facecolor=color + (0.15,))  # Same color, very transparent fill
            axes[idx].add_patch(rect)
            
            if category_names is not None:
                label_text = category_names.get(label.item(), f'Class {label.item()}')
                axes[idx].text(x1, y1-5, label_text,
                             color='white',
                             fontsize=8,
                             bbox=dict(facecolor=color, alpha=0.7))
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_grid_by_class(dataset, category_names, samples_per_class=5, output_file_prefix='class_grid', min_visibility=0.5):
    """
    Visualize a grid where each row is a class and each column is a random sample from that class.
    Bounding boxes are drawn in a fixed color (red) for better visibility, with transparent fill.
    Split into two separate images for better readability.
    Ensures no duplicate images are shown in each row.
    """
    import torch
    class_ids = sorted(category_names.keys())
    n_classes = len(class_ids)
    # Use a fixed color (red) for all bounding boxes
    box_color = (1.0, 0.0, 0.0)  # Red in RGB
    
    # For each class, collect sample indices containing at least one instance of that class
    class_to_indices = {cid: [] for cid in class_ids}
    visible_indices = set(find_visible_box_indices(dataset, min_visibility))
    
    for idx in range(len(dataset)):
        if idx not in visible_indices:
            continue
        labels = dataset[idx][1]['labels']
        for cid in class_ids:
            if (labels == cid).sum() > 0 if isinstance(labels, torch.Tensor) else labels.count(cid) > 0:
                class_to_indices[cid].append(idx)
    
    def create_grid(used_indices=None):
        if used_indices is None:
            used_indices = set()
        
        fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(samples_per_class*4, n_classes*4))
        if n_classes == 1:
            axes = np.expand_dims(axes, 0)
        if samples_per_class == 1:
            axes = np.expand_dims(axes, 1)
        
        current_used_indices = set(used_indices)
        
        for row, cid in enumerate(class_ids):
            indices = [idx for idx in class_to_indices[cid] if idx not in current_used_indices]
            if len(indices) == 0:
                for col in range(samples_per_class):
                    axes[row, col].axis('off')
                continue
            
            # Get unique samples, avoiding any previously used indices
            if len(indices) < samples_per_class:
                print(f"Warning: Only found {len(indices)} unique images for class {category_names[cid]}. Using all available.")
                chosen = indices
            else:
                chosen = random.sample(indices, samples_per_class)
            
            current_used_indices.update(chosen)
            
            for col, idx in enumerate(chosen):
                image, targets = dataset[idx]
                image = ensure_numpy_hwc(image)
                axes[row, col].imshow(image)
                # Draw only boxes for this class
                for box, label in zip(targets['boxes'], targets['labels']):
                    if (label.item() if isinstance(label, torch.Tensor) else label) == cid:
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        rect = Rectangle((x1, y1), width, height,
                                         linewidth=2,
                                         edgecolor=box_color,
                                         facecolor=box_color + (0.15,))  # Same color, very transparent fill
                        axes[row, col].add_patch(rect)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if col == 0:
                    axes[row, col].set_ylabel(category_names[cid], fontsize=16, rotation=0, labelpad=80, va='center')
        
        plt.tight_layout()
        return fig, current_used_indices
    
    # Create first grid
    fig1, used_indices = create_grid()
    plt.savefig(f'{output_file_prefix}_1.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create second grid, avoiding images used in the first grid
    fig2, _ = create_grid(used_indices)
    plt.savefig(f'{output_file_prefix}_2.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Define transformations
    transform = transforms.ToTensor()  # Only ToTensor for visualization

    # Create dataset
    data_dir = 'data/20250507_NORD_FKB_Som_Korrigert'
    ann_file = os.path.join(data_dir, 'coco_dataset.json')
    dataset = COCOObjectDetectionDataset(data_dir, ann_file, transform=transform)
    
    # Build category names from annotation file
    category_names = build_category_names(ann_file)
    
    # Visualize a batch of images
    fig = visualize_batch(dataset, num_samples=4, category_names=category_names)
    plt.savefig('dataset_samples.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Visualize a single image (prefer uncluttered)
    uncluttered_indices = find_uncluttered_indices(dataset, max_boxes=3)
    if uncluttered_indices:
        idx = random.choice(uncluttered_indices)
    else:
        idx = random.randint(0, len(dataset)-1)
    image, targets = dataset[idx]
    fig = visualize_image(image, targets['boxes'], targets['labels'], 
                         category_names=category_names)
    plt.savefig('single_sample.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Visualize a grid by class
    visualize_grid_by_class(dataset, category_names, samples_per_class=5, output_file_prefix='class_grid') 