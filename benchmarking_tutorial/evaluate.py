from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np
import torch
import os
import cv2

def prepare_coco_format(predictions, image_ids):
    """
    Prepare predictions in COCO format.
    """
    coco_predictions = []
    for img_id, preds in zip(image_ids, predictions):
        for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            coco_predictions.append({
                'image_id': img_id,
                'category_id': label,
                'bbox': [x_min.item(), y_min.item(), width.item(), height.item()],
                'score': score.item(),
            })
    return coco_predictions

def evaluate_coco(dataloader, model, coco_gt, device):
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

    # Prepare predictions in COCO format
    coco_predictions = prepare_coco_format(all_predictions, image_ids)

    # Load predictions into COCO
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def prepare_coco_segmentation_format(predictions, image_ids):
    """
    Prepare segmentation predictions in COCO format.
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
            rle = maskUtils.encode(np.asfortranarray(class_mask))
            rle['counts'] = rle['counts'].decode('ascii')

            coco_predictions.append({
                'id': ann_id,  # Add unique ID for each annotation
                'image_id': img_id,
                'category_id': int(class_id),
                'segmentation': rle,
                'score': 1.0,  # Dummy score for demonstration
                'area': float(np.sum(class_mask)),  # Add area field
                'iscrowd': 0  # Add iscrowd field
            })
            ann_id += 1  # Increment ID for next annotation
            
    return coco_predictions


def evaluate_coco_segmentation(dataloader, model, coco_gt, device):
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

    # Prepare predictions in COCO format
    coco_predictions = prepare_coco_segmentation_format(all_predictions, image_ids)

    # Create a new COCO dataset for predictions
    coco_dt = COCO()
    coco_dt.dataset = {
        'images': [],
        'annotations': [],
        'categories': coco_gt.loadCats(coco_gt.getCatIds())
    }
    
    # Add images
    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        coco_dt.dataset['images'].append(img_info)
    
    # Add annotations
    for pred in coco_predictions:
        coco_dt.dataset['annotations'].append(pred)
    
    coco_dt.createIndex()

    # Create a new COCO dataset for ground truth with only annotations that have segmentation
    coco_gt_seg = COCO()
    coco_gt_seg.dataset = {
        'images': [],
        'annotations': [],
        'categories': coco_gt.loadCats(coco_gt.getCatIds())
    }

    # Add images
    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        coco_gt_seg.dataset['images'].append(img_info)

    # Add only annotations that have segmentation
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_ids)):
        if 'segmentation' in ann:
            coco_gt_seg.dataset['annotations'].append(ann)

    coco_gt_seg.createIndex()

    # Print statistics
    total_anns = len(coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_ids)))
    seg_anns = len(coco_gt_seg.dataset['annotations'])
    print(f"\nSegmentation Evaluation Statistics:")
    print(f"Total annotations: {total_anns}")
    print(f"Annotations with segmentation: {seg_anns} ({seg_anns/total_anns*100:.1f}%)")

    # Initialize COCOeval with the filtered ground truth
    coco_eval = COCOeval(coco_gt_seg, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()