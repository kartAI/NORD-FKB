# Custom Model Integration Guide for COCO Segmentation Evaluation

This guide explains how to integrate your own segmentation model with our COCO evaluation tools. It covers the complete pipeline from model output to COCO format conversion and evaluation.

## Dataset Structure

The dataset follows the COCO format with the following structure:
```
dataset_root/
├── Image_rgb/          # RGB images
├── Mask/               # Individual mask files
└── coco_dataset.json   # COCO format annotations
```

### Mask File Naming Convention
Individual mask files are named using the following format:
```
{image_name}-{annotation_index}-{category_name}.tif
```
For example: `image_001-0-building.tif`

The annotation index is zero-based and corresponds to the order of annotations in the COCO JSON file for each image.

## Model Requirements

Your model should output segmentation masks in one of these formats:

1. **Class-wise Masks**: A tensor of shape `[batch_size, num_classes, height, width]` where each channel represents a binary mask for a specific class.
   - Each channel should be a binary mask (0 or 1)
   - Channel index corresponds to class ID
   - Background is typically class 0

2. **Instance Masks**: A tensor of shape `[batch_size, height, width]` where each pixel value represents the class ID.
   - Each pixel value should be an integer corresponding to the class ID
   - 0 typically represents background
   - Values should match the category IDs in your COCO dataset

## Integration Steps

### 1. Create Your Model Class

Your model should inherit from `torch.nn.Module` and implement the forward pass. Here's a template:

```python
import torch
import torch.nn as nn

class YourSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your model architecture here
        self.num_classes = num_classes

    def forward(self, x):
        # Your forward pass here
        # Should return a tensor of shape [batch_size, num_classes, height, width]
        # or [batch_size, height, width] for instance masks
        return output
```

#### Example Model Output
```python
# For class-wise masks (format 1):
output = torch.zeros((batch_size, num_classes, height, width))
# Each channel is a binary mask for a specific class

# For instance masks (format 2):
output = torch.zeros((batch_size, height, width), dtype=torch.long)
# Each pixel value is the class ID
```

### 2. Prepare Model Output for Evaluation

The model output needs to be converted to COCO format for evaluation. Here's how to do it:

```python
from pycocotools import mask as maskUtils
import numpy as np

def prepare_coco_segmentation_format(predictions, image_ids):
    """
    Convert model predictions to COCO format.
    
    Args:
        predictions: List of model outputs (masks)
        image_ids: List of corresponding image IDs
    
    Returns:
        List of predictions in COCO format
    """
    coco_predictions = []
    ann_id = 1  # Start with ID 1 for annotations
    
    for img_id, pred_mask in zip(image_ids, predictions):
        # Convert PyTorch tensor to NumPy array
        pred_mask_np = pred_mask.cpu().numpy()
        
        # Get unique class IDs from the mask
        unique_classes = np.unique(pred_mask_np)
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
                
            # Create binary mask for this class
            class_mask = (pred_mask_np == class_id).astype(np.uint8)
            
            # Convert to RLE format
            rle = maskUtils.encode(np.asfortranarray(class_mask))
            rle['counts'] = rle['counts'].decode('ascii')

            # Create COCO format prediction
            coco_predictions.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(class_id),
                'segmentation': rle,
                'score': 1.0,  # Replace with actual confidence score if available
                'area': float(np.sum(class_mask)),
                'iscrowd': 0
            })
            ann_id += 1
            
    return coco_predictions
```

#### Handling Class-wise Masks
If your model outputs class-wise masks, you'll need to modify the conversion:

```python
# For class-wise masks (format 1)
for class_id in range(1, pred_mask_np.shape[0]):  # Skip background (class 0)
    class_mask = pred_mask_np[class_id].astype(np.uint8)
    if np.any(class_mask):  # Only process if mask is not empty
        rle = maskUtils.encode(np.asfortranarray(class_mask))
        rle['counts'] = rle['counts'].decode('ascii')
        
        coco_predictions.append({
            'id': ann_id,
            'image_id': img_id,
            'category_id': class_id,
            'segmentation': rle,
            'score': 1.0,
            'area': float(np.sum(class_mask)),
            'iscrowd': 0
        })
        ann_id += 1
```

### 3. Evaluation Process

Here's how to evaluate your model:

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch

def evaluate_model(model, dataloader, coco_gt, device):
    """
    Evaluate model performance using COCO metrics.
    
    Args:
        model: Your segmentation model
        dataloader: DataLoader for the dataset
        coco_gt: COCO ground truth object
        device: Device to run evaluation on
    """
    model.eval()
    all_predictions = []
    image_ids = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            
            # Collect predictions and image ids
            all_predictions.extend(outputs)
            image_ids.extend([target['image_id'] for target in targets])

    # Convert predictions to COCO format
    coco_predictions = prepare_coco_segmentation_format(all_predictions, image_ids)

    # Create COCO dataset for predictions
    coco_dt = COCO()
    coco_dt.dataset = {
        'images': [],
        'annotations': [],
        'categories': coco_gt.loadCats(coco_gt.getCatIds())
    }
    
    # Add images and annotations
    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        coco_dt.dataset['images'].append(img_info)
    
    for pred in coco_predictions:
        coco_dt.dataset['annotations'].append(pred)
    
    coco_dt.createIndex()

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
```

## Example Usage

Here's a complete example of how to use the evaluation tools:

```python
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Initialize your model
num_classes = 5  # Including background
model = YourSegmentationModel(num_classes=num_classes)
model.to(device)

# Load dataset
data_dir = 'path/to/dataset'
ann_file = 'path/to/coco_dataset.json'
dataset = COCOSegmentationDataset(data_dir, ann_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

# Load ground truth
coco_gt = COCO(ann_file)

# Run evaluation
evaluate_model(model, dataloader, coco_gt, device)
```

## Expected Output Format

The evaluation will output standard COCO metrics:
- Average Precision (AP) at different IoU thresholds:
  - AP@[IoU=0.50:0.95] (primary metric)
  - AP@[IoU=0.50]
  - AP@[IoU=0.75]
- Average Recall (AR) at different detection thresholds:
  - AR@[maxDets=1]
  - AR@[maxDets=10]
  - AR@[maxDets=100]
- Metrics for different object sizes:
  - AP/AR for small objects (area < 32²)
  - AP/AR for medium objects (32² < area < 96²)
  - AP/AR for large objects (area > 96²)

## Troubleshooting

1. **Zero Annotations with Segmentation**: 
   - Check if your model's output format matches the expected format
   - Verify that class IDs in predictions match the COCO dataset categories
   - Ensure masks are properly converted to RLE format
   - Check if masks are binary (0 or 1) before conversion
   - Verify that the mask files exist and are readable

2. **Low Performance**:
   - Verify that your model's output is properly normalized
   - Check if class IDs are correctly mapped
   - Ensure masks are binary (0 or 1) before conversion to RLE
   - Verify that the ground truth annotations are correct
   - Check if the evaluation metrics are appropriate for your use case

3. **Memory Issues**:
   - Reduce batch size
   - Process predictions in smaller chunks
   - Use GPU if available
   - Clear GPU memory between batches
   - Use mixed precision training if applicable

4. **Slow Evaluation**:
   - Increase number of worker threads in DataLoader
   - Process predictions in parallel if possible
   - Use GPU for model inference
   - Optimize mask conversion process

## Additional Notes

- The evaluation uses the standard COCO metrics for instance segmentation
- Make sure your model's output matches the ground truth format
- Consider adding confidence scores to your predictions for better evaluation
- The evaluation process can be memory-intensive for large datasets
- You can customize the evaluation parameters in COCOeval:
  ```python
  coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
  coco_eval.params.iouThrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
  coco_eval.params.maxDets = [1, 10, 100]
  ```

## Best Practices

1. **Data Preparation**:
   - Ensure consistent image and mask sizes
   - Normalize images appropriately
   - Verify mask file naming convention
   - Check COCO JSON format compliance

2. **Model Output**:
   - Use appropriate activation functions (e.g., sigmoid for binary masks)
   - Ensure proper class ID mapping
   - Include confidence scores if available
   - Handle multi-class predictions correctly

3. **Evaluation**:
   - Use appropriate batch size for your hardware
   - Monitor memory usage during evaluation
   - Save evaluation results for comparison
   - Consider using multiple evaluation metrics

4. **Performance Optimization**:
   - Use GPU acceleration when available
   - Implement efficient mask conversion
   - Optimize data loading pipeline
   - Consider using mixed precision 