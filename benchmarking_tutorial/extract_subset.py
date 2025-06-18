#!/usr/bin/env python3
"""
Script to extract a small subset of the NORD-FKB dataset for testing and development.
This script creates a smaller version of the dataset with a specified number of images.
"""

import os
import json
import shutil
import random
from pathlib import Path
import argparse

def extract_dataset_subset(
    source_dir, 
    output_dir, 
    num_images=10, 
    random_seed=42,
    include_masks=True
):
    """
    Extract a subset of the dataset.
    
    Args:
        source_dir (str): Path to the source dataset directory
        output_dir (str): Path to the output directory for the subset
        num_images (int): Number of images to include in the subset
        random_seed (int): Random seed for reproducible sampling
        include_masks (bool): Whether to include mask files
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "Image_rgb").mkdir(exist_ok=True)
    if include_masks:
        (output_path / "Mask").mkdir(exist_ok=True)
    
    # Load COCO annotations
    coco_file = Path(source_dir) / "coco_dataset.json"
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Original dataset has {len(coco_data['images'])} images")
    print(f"Original dataset has {len(coco_data['annotations'])} annotations")
    print(f"Original dataset has {len(coco_data['categories'])} categories")
    
    # Randomly sample images
    all_image_ids = [img['id'] for img in coco_data['images']]
    selected_image_ids = random.sample(all_image_ids, min(num_images, len(all_image_ids)))
    
    print(f"Selected {len(selected_image_ids)} images for subset")
    
    # Filter images and annotations
    subset_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
    subset_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids]
    
    # Get unique categories from selected annotations
    selected_category_ids = set(ann['category_id'] for ann in subset_annotations)
    subset_categories = [cat for cat in coco_data['categories'] if cat['id'] in selected_category_ids]
    
    # Create subset COCO data
    subset_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': subset_categories,
        'images': subset_images,
        'annotations': subset_annotations
    }
    
    # Copy selected image files
    source_image_dir = Path(source_dir) / "Image_rgb"
    output_image_dir = output_path / "Image_rgb"
    
    print("Copying image files...")
    for img in subset_images:
        # Handle both cases: with and without Image_rgb/ prefix
        filename = img['file_name']
        if filename.startswith('Image_rgb/'):
            filename = filename.replace('Image_rgb/', '')
        
        source_file = source_image_dir / filename
        output_file = output_image_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, output_file)
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: Image file not found: {filename}")
    
    # Copy mask files if requested
    if include_masks:
        source_mask_dir = Path(source_dir) / "Mask"
        output_mask_dir = output_path / "Mask"
        
        if source_mask_dir.exists():
            print("Copying mask files...")
            # Get all mask files that correspond to our selected images
            selected_filenames = set()
            for img in subset_images:
                filename = img['file_name']
                if filename.startswith('Image_rgb/'):
                    filename = filename.replace('Image_rgb/', '')
                selected_filenames.add(filename.replace('.tif', ''))
            
            # Find mask files that start with our selected image names
            mask_files_copied = 0
            for mask_file in source_mask_dir.glob('*.tif'):
                mask_name = mask_file.stem  # filename without extension
                # Check if this mask corresponds to any of our selected images
                for selected_name in selected_filenames:
                    if mask_name.startswith(selected_name + '-'):
                        shutil.copy2(mask_file, output_mask_dir / mask_file.name)
                        print(f"  Copied mask: {mask_file.name}")
                        mask_files_copied += 1
                        break
            
            if mask_files_copied == 0:
                print("  Warning: No mask files found for selected images")
            else:
                print(f"  Total mask files copied: {mask_files_copied}")
        else:
            print("Warning: Mask directory not found, skipping mask files")
    
    # Save subset COCO annotations
    subset_coco_file = output_path / "coco_dataset.json"
    with open(subset_coco_file, 'w') as f:
        json.dump(subset_coco_data, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUBSET CREATION SUMMARY")
    print("="*50)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images included: {len(subset_images)}")
    print(f"Annotations included: {len(subset_annotations)}")
    print(f"Categories included: {len(subset_categories)}")
    print(f"Categories: {[cat['name'] for cat in subset_categories]}")
    
    # Calculate some statistics
    annotations_per_image = len(subset_annotations) / len(subset_images)
    print(f"Average annotations per image: {annotations_per_image:.2f}")
    
    # Category distribution
    category_counts = {}
    for ann in subset_annotations:
        cat_id = ann['category_id']
        cat_name = next(cat['name'] for cat in subset_categories if cat['id'] == cat_id)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    print("\nCategory distribution in subset:")
    for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name}: {count} instances")
    
    print(f"\nSubset saved to: {output_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Extract a subset of the NORD-FKB dataset")
    parser.add_argument(
        "--source-dir", 
        default="data/20250507_NORD_FKB_Som_Korrigert",
        help="Source dataset directory (default: data/20250507_NORD_FKB_Som_Korrigert)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/NORD_FKB_subset",
        help="Output directory for the subset (default: data/NORD_FKB_subset)"
    )
    parser.add_argument(
        "--num-images", 
        type=int, 
        default=10,
        help="Number of images to include in the subset (default: 10)"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--no-masks", 
        action="store_true",
        help="Skip copying mask files"
    )
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist")
        return
    
    # Extract subset
    extract_dataset_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        random_seed=args.random_seed,
        include_masks=not args.no_masks
    )

if __name__ == "__main__":
    main() 