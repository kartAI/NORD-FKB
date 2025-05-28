import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def unnormalize_image(tensor, mean, std):
    """
    Un-normalize a tensor image to its original form.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_image(image_tensor, mask_tensor=None):
    """
    Visualize a single image tensor with an optional mask overlay.
    """
    # Un-normalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = unnormalize_image(image_tensor.clone(), mean, std)
    
    # Convert to numpy and transpose to (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # Clip values to [0, 1] range for display
    image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    
    if mask_tensor is not None:
        # Convert mask to numpy
        mask = mask_tensor.cpu().numpy()
        
        # Create a color mask
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = [1, 0, 0]  # Red color for mask
        
        # Overlay mask on image
        plt.imshow(color_mask, alpha=0.5)  # Adjust alpha for transparency

    plt.axis('off')
    plt.show()


def visualize_image_with_bboxes(image_tensor, bboxes_and_labels=None):
    """
    Visualize a single image tensor with optional bounding boxes.
    """

    if bboxes_and_labels is not None:
        bboxes = bboxes_and_labels["boxes"]
        labels = bboxes_and_labels["labels"]
    else:
        bboxes = None
        labels = None

    # Un-normalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = unnormalize_image(image_tensor.clone(), mean, std)
    
    # Convert to numpy and transpose to (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # Clip values to [0, 1] range for display
    image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    ax = plt.gca()

    if bboxes is not None:
        # Iterate over each bounding box
        for i, bbox in enumerate(bboxes):
            # Generate a random color
            color = [random.random() for _ in range(3)]
            
            # Create a rectangle patch with a random color
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor=color, facecolor='none')
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
            # Add category name as text with the same color
            if labels is not None:
                plt.text(bbox[0], bbox[1] - 10, str(labels[i].item()), color=color, fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.show()