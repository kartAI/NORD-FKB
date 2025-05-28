import json
import os
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

def add_segmentation_to_annotations(json_path, data_dir):
    # Load the existing annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create mask directory path
    mask_dir = os.path.join(data_dir, 'Mask')

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        if ann['image_id'] not in annotations_by_image:
            annotations_by_image[ann['image_id']] = []
        annotations_by_image[ann['image_id']].append(ann)

    # Add segmentation information to each annotation
    for img_id, anns in tqdm(annotations_by_image.items(), desc="Adding segmentation masks"):
        # Get image info
        img_info = next(img for img in data['images'] if img['id'] == img_id)
        
        # Process each annotation for this image
        for ann_idx, ann in enumerate(anns):
            # Get category name
            category_name = next(cat['name'] for cat in data['categories'] if cat['id'] == ann['category_id'])
            
            # Construct mask filename using annotation index
            mask_filename = f"{img_info['file_name'].replace('.tif', '')}-{ann_idx}-{category_name}.tif".replace("Image_rgb/", "")
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Load and encode mask
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Convert mask to RLE format
                    rle = maskUtils.encode(np.asfortranarray(mask > 0))
                    rle['counts'] = rle['counts'].decode('ascii')
                    
                    # Add segmentation and area information
                    ann['segmentation'] = rle
                    ann['area'] = float(np.sum(mask > 0))
                    ann['iscrowd'] = 0
                else:
                    print(f"Warning: Could not read mask file {mask_path}")
            else:
                print(f"Warning: Mask file not found: {mask_path}")

    # Save the updated annotations
    output_path = json_path.replace('.json', '_with_segmentation.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated annotations saved to {output_path}")

if __name__ == "__main__":
    # Path to your COCO dataset JSON file and data directory
    json_path = '20250507_NORD_FKB_Som_Korrigert/coco_dataset.json'
    data_dir = '20250507_NORD_FKB_Som_Korrigert'
    
    add_segmentation_to_annotations(json_path, data_dir) 