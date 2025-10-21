#!/usr/bin/env python3
"""
Script to combine multiple COCO-format datasets in benchmarking_tutorial/data/ into one combined dataset.
- Handles ID and filename collisions
- Merges categories by name
- Copies images and masks
- Outputs to benchmarking_tutorial/data/combined_dataset/
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'combined_dataset'
IMAGE_DIR_NAME = 'Image_rgb'
MASK_DIR_NAME = 'Mask'
COCO_FILENAME = 'coco_dataset.json'


def find_datasets(data_dir):
    """Find all dataset folders in data_dir that contain a coco_dataset.json file."""
    datasets = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and (subdir / COCO_FILENAME).exists():
            datasets.append(subdir)
    return datasets


def merge_categories(all_categories):
    """Merge categories by name, assign new unique IDs."""
    name_to_id = {}
    merged = []
    next_id = 1
    for cats in all_categories:
        for cat in cats:
            name = cat['name']
            if name not in name_to_id:
                new_cat = cat.copy()
                new_cat['id'] = next_id
                name_to_id[name] = next_id
                merged.append(new_cat)
                next_id += 1
    return merged, name_to_id


def main():
    datasets = find_datasets(DATA_DIR)
    if not datasets:
        print(f"No datasets found in {DATA_DIR}")
        return
    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / IMAGE_DIR_NAME).mkdir(exist_ok=True)
    (OUTPUT_DIR / MASK_DIR_NAME).mkdir(exist_ok=True)

    all_images = []
    all_annotations = []
    all_categories = []
    all_licenses = []
    all_info = []

    image_id_map = {}  # (dataset_idx, old_image_id) -> new_image_id
    ann_id_map = {}    # (dataset_idx, old_ann_id) -> new_ann_id
    filename_map = {}  # old filename -> new filename (if renamed)

    next_image_id = 1
    next_ann_id = 1

    # First, collect all categories for merging
    for dataset in datasets:
        with open(dataset / COCO_FILENAME, 'r') as f:
            coco = json.load(f)
        all_categories.append(coco['categories'])

    merged_categories, name_to_new_cat_id = merge_categories(all_categories)

    # Now process each dataset
    for idx, dataset in enumerate(datasets):
        with open(dataset / COCO_FILENAME, 'r') as f:
            coco = json.load(f)
        # Merge info and licenses (just take the first for simplicity)
        if idx == 0:
            all_info = coco.get('info', {})
            all_licenses = coco.get('licenses', [])

        # Map old category IDs to new
        old_catid_to_new = {cat['id']: name_to_new_cat_id[cat['name']] for cat in coco['categories']}

        # Copy images
        for img in coco['images']:
            old_id = img['id']
            old_filename = img['file_name']
            # Ensure unique filenames in combined set
            new_filename = f"{idx}_{old_filename.replace('/', '_')}"
            filename_map[(idx, old_filename)] = new_filename
            new_img = img.copy()
            new_img['id'] = next_image_id
            new_img['file_name'] = f"{IMAGE_DIR_NAME}/{new_filename}"
            image_id_map[(idx, old_id)] = next_image_id
            all_images.append(new_img)
            # Copy image file
            src_img = dataset / IMAGE_DIR_NAME / old_filename.split('/')[-1]
            dst_img = OUTPUT_DIR / IMAGE_DIR_NAME / new_filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Image file not found: {src_img}")
            next_image_id += 1

        # Copy masks (if present)
        mask_dir = dataset / MASK_DIR_NAME
        if mask_dir.exists():
            for mask_file in mask_dir.glob('*.tif'):
                # Prefix mask filename with dataset index to avoid collisions
                new_mask_name = f"{idx}_{mask_file.name}"
                dst_mask = OUTPUT_DIR / MASK_DIR_NAME / new_mask_name
                shutil.copy2(mask_file, dst_mask)

        # Remap annotations
        for ann in coco['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = image_id_map[(idx, ann['image_id'])]
            new_ann['category_id'] = old_catid_to_new[ann['category_id']]
            all_annotations.append(new_ann)
            next_ann_id += 1

    # Write combined COCO file
    combined_coco = {
        'info': all_info,
        'licenses': all_licenses,
        'categories': merged_categories,
        'images': all_images,
        'annotations': all_annotations
    }
    with open(OUTPUT_DIR / COCO_FILENAME, 'w') as f:
        json.dump(combined_coco, f, indent=2)
    print(f"Combined dataset written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 