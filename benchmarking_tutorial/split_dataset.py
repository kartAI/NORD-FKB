#!/usr/bin/env python3
"""
Script to split the NORD-FKB dataset into train and validation sets.
This script ensures approximately balanced class distributions between the splits.
"""

import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict, Counter

def calculate_class_distribution(annotations, categories):
    """Calculate the distribution of classes in the annotations."""
    category_counts = defaultdict(int)
    for ann in annotations:
        category_counts[ann['category_id']] += 1
    return category_counts

def get_images_by_category(annotations):
    """Group images by the categories they contain."""
    images_by_category = defaultdict(set)
    for ann in annotations:
        images_by_category[ann['category_id']].add(ann['image_id'])
    return images_by_category

def split_dataset_balanced(
    source_dir,
    train_output_dir,
    val_output_dir,
    train_ratio=0.8,
    random_seed=42,
    include_masks=True,
    max_imbalance_ratio=2.0
):
    """
    Split dataset into train and validation sets with balanced class distributions.
    
    Args:
        source_dir (str): Path to the source dataset directory
        train_output_dir (str): Path to the training set output directory
        val_output_dir (str): Path to the validation set output directory
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        random_seed (int): Random seed for reproducible splitting
        include_masks (bool): Whether to include mask files
        max_imbalance_ratio (float): Maximum allowed ratio between train/val class distributions
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load COCO annotations
    coco_file = Path(source_dir) / "coco_dataset.json"
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Original dataset has {len(coco_data['images'])} images")
    print(f"Original dataset has {len(coco_data['annotations'])} annotations")
    print(f"Original dataset has {len(coco_data['categories'])} categories")
    
    # Calculate original class distribution
    original_dist = calculate_class_distribution(coco_data['annotations'], coco_data['categories'])
    print("\nOriginal class distribution:")
    for cat_id, count in sorted(original_dist.items()):
        cat_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == cat_id)
        print(f"  {cat_name}: {count} instances")
    
    # Strategy: Start with a simple random split, then balance by moving images
    all_image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(all_image_ids)
    
    # Initial split
    split_point = int(len(all_image_ids) * train_ratio)
    train_image_ids = set(all_image_ids[:split_point])
    val_image_ids = set(all_image_ids[split_point:])
    
    print(f"\nInitial split: {len(train_image_ids)} train, {len(val_image_ids)} val")
    
    # Group images by categories they contain
    images_by_category = get_images_by_category(coco_data['annotations'])
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Balance the split by ensuring each category has representatives in both sets
    max_iterations = 10
    for iteration in range(max_iterations):
        print(f"\nBalancing iteration {iteration + 1}...")
        
        # Check class distribution in current split
        train_data = create_split_dataset(coco_data, list(train_image_ids), "train")
        val_data = create_split_dataset(coco_data, list(val_image_ids), "validation")
        
        train_dist = calculate_class_distribution(train_data['annotations'], train_data['categories'])
        val_dist = calculate_class_distribution(val_data['annotations'], val_data['categories'])
        
        # Find categories that are missing from validation set
        missing_in_val = []
        for cat_id in original_dist.keys():
            if val_dist.get(cat_id, 0) == 0 and train_dist.get(cat_id, 0) > 0:
                missing_in_val.append(cat_id)
        
        if not missing_in_val:
            print("All categories are represented in both sets. Balancing complete.")
            break
        
        # Move some images from train to val for missing categories
        moved_count = 0
        for cat_id in missing_in_val:
            cat_name = cat_id_to_name[cat_id]
            cat_images = images_by_category[cat_id]
            train_cat_images = cat_images.intersection(train_image_ids)
            
            if train_cat_images:
                # Move one image from train to val for this category
                image_to_move = random.choice(list(train_cat_images))
                train_image_ids.remove(image_to_move)
                val_image_ids.add(image_to_move)
                moved_count += 1
                print(f"  Moved image for category '{cat_name}' from train to val")
        
        if moved_count == 0:
            print("No more images can be moved. Balancing stopped.")
            break
    
    # Finalize image sets
    train_image_ids = list(train_image_ids)
    val_image_ids = list(val_image_ids)
    
    print(f"\nFinal split: {len(train_image_ids)} train images, {len(val_image_ids)} val images")
    print(f"Split ratio: {len(train_image_ids)/(len(train_image_ids) + len(val_image_ids)):.2%} train, {len(val_image_ids)/(len(train_image_ids) + len(val_image_ids)):.2%} val")
    
    # Create train and validation datasets
    train_data = create_split_dataset(coco_data, train_image_ids, "train")
    val_data = create_split_dataset(coco_data, val_image_ids, "validation")
    
    # Check class distribution balance
    train_dist = calculate_class_distribution(train_data['annotations'], train_data['categories'])
    val_dist = calculate_class_distribution(val_data['annotations'], val_data['categories'])
    
    print("\nClass distribution after split:")
    print("Category".ljust(20) + "Train".ljust(10) + "Val".ljust(10) + "Ratio".ljust(10))
    print("-" * 50)
    
    imbalances = []
    for cat_id in sorted(original_dist.keys()):
        cat_name = cat_id_to_name[cat_id]
        train_count = train_dist.get(cat_id, 0)
        val_count = val_dist.get(cat_id, 0)
        
        if val_count > 0:
            ratio = train_count / val_count
        else:
            ratio = float('inf')
        
        imbalances.append(ratio)
        print(f"{cat_name[:19].ljust(20)}{str(train_count).ljust(10)}{str(val_count).ljust(10)}{f'{ratio:.2f}'.ljust(10)}")
    
    max_imbalance = max(imbalances) if imbalances else 0
    print(f"\nMaximum train/val ratio: {max_imbalance:.2f}")
    
    if max_imbalance > max_imbalance_ratio:
        print(f"Warning: Maximum imbalance ratio ({max_imbalance:.2f}) exceeds threshold ({max_imbalance_ratio})")
    
    # Save datasets and copy files
    save_split_dataset(train_data, train_output_dir, source_dir, include_masks, "train")
    save_split_dataset(val_data, val_output_dir, source_dir, include_masks, "validation")
    
    print(f"\nDatasets saved to:")
    print(f"  Train: {train_output_dir}")
    print(f"  Validation: {val_output_dir}")

def create_split_dataset(coco_data, image_ids, split_name):
    """Create a dataset split with the given image IDs."""
    # Filter images and annotations
    split_images = [img for img in coco_data['images'] if img['id'] in image_ids]
    split_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    # Get unique categories from selected annotations
    selected_category_ids = set(ann['category_id'] for ann in split_annotations)
    split_categories = [cat for cat in coco_data['categories'] if cat['id'] in selected_category_ids]
    
    # Create split COCO data
    split_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': split_categories,
        'images': split_images,
        'annotations': split_annotations
    }
    
    return split_coco_data

def save_split_dataset(split_data, output_dir, source_dir, include_masks, split_name):
    """Save a dataset split to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "Image_rgb").mkdir(exist_ok=True)
    if include_masks:
        (output_path / "Mask").mkdir(exist_ok=True)
    
    # Copy image files
    source_image_dir = Path(source_dir) / "Image_rgb"
    output_image_dir = output_path / "Image_rgb"
    
    print(f"\nCopying {split_name} image files...")
    for img in split_data['images']:
        filename = img['file_name']
        if filename.startswith('Image_rgb/'):
            filename = filename.replace('Image_rgb/', '')
        
        source_file = source_image_dir / filename
        output_file = output_image_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, output_file)
        else:
            print(f"  Warning: Image file not found: {filename}")
    
    # Copy mask files if requested
    if include_masks:
        source_mask_dir = Path(source_dir) / "Mask"
        output_mask_dir = output_path / "Mask"
        
        if source_mask_dir.exists():
            print(f"Copying {split_name} mask files...")
            selected_filenames = set()
            for img in split_data['images']:
                filename = img['file_name']
                if filename.startswith('Image_rgb/'):
                    filename = filename.replace('Image_rgb/', '')
                # Extract the tile identifier (e.g., "2_tile-57-131" from "2_Image_rgb_tile-57-131.tif")
                tile_id = filename.replace('Image_rgb_', '').replace('.tif', '')
                selected_filenames.add(tile_id)
            
            mask_files_copied = 0
            for mask_file in source_mask_dir.glob('*.tif'):
                mask_name = mask_file.stem
                # Check if this mask belongs to any selected image
                for selected_name in selected_filenames:
                    if mask_name.startswith(selected_name + '-'):
                        shutil.copy2(mask_file, output_mask_dir / mask_file.name)
                        mask_files_copied += 1
                        break
            
            print(f"  Total mask files copied: {mask_files_copied}")
        else:
            print("Warning: Mask directory not found, skipping mask files")
    
    # Save COCO annotations
    coco_file = output_path / "coco_dataset.json"
    with open(coco_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"{split_name.capitalize()} dataset saved with {len(split_data['images'])} images and {len(split_data['annotations'])} annotations")

def main():
    parser = argparse.ArgumentParser(description="Split the NORD-FKB dataset into train and validation sets")
    parser.add_argument(
        "--source-dir", 
        default="data/combined_dataset",
        help="Source dataset directory (default: data/combined_dataset)"
    )
    parser.add_argument(
        "--train-output-dir", 
        default="data/NORD_FKB_train",
        help="Output directory for training set (default: data/NORD_FKB_train)"
    )
    parser.add_argument(
        "--val-output-dir", 
        default="data/NORD_FKB_val",
        help="Output directory for validation set (default: data/NORD_FKB_val)"
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.8,
        help="Ratio of data to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible splitting (default: 42)"
    )
    parser.add_argument(
        "--no-masks", 
        action="store_true",
        help="Skip copying mask files"
    )
    parser.add_argument(
        "--max-imbalance-ratio", 
        type=float, 
        default=2.0,
        help="Maximum allowed ratio between train/val class distributions (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Use relative paths from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, args.source_dir)
    train_output_dir = os.path.join(script_dir, args.train_output_dir)
    val_output_dir = os.path.join(script_dir, args.val_output_dir)
    
    # Validate arguments
    if not 0 < args.train_ratio < 1:
        print("Error: train_ratio must be between 0 and 1")
        return
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        print("Please run combine_datasets.py first to create the combined dataset.")
        return
    
    # Check if COCO file exists
    coco_file = os.path.join(source_dir, "coco_dataset.json")
    if not os.path.exists(coco_file):
        print(f"Error: COCO dataset file not found at {coco_file}")
        print("Please run combine_datasets.py first to create the combined dataset.")
        return
    
    # Split dataset
    split_dataset_balanced(
        source_dir=source_dir,
        train_output_dir=train_output_dir,
        val_output_dir=val_output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        include_masks=not args.no_masks,
        max_imbalance_ratio=args.max_imbalance_ratio
    )

if __name__ == "__main__":
    main() 