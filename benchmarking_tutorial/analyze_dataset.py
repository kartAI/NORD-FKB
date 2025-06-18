import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from collections import Counter, defaultdict
import pandas as pd
import argparse

def analyze_dataset(ann_file):
    """Analyze COCO format dataset and return statistics."""
    # Load COCO dataset
    coco = COCO(ann_file)
    
    # Get basic statistics
    n_images = len(coco.getImgIds())
    n_categories = len(coco.getCatIds())
    n_annotations = len(coco.getAnnIds())
    
    # Get category information
    cats = coco.loadCats(coco.getCatIds())
    category_names = {cat['id']: cat['name'] for cat in cats}
    
    # Count instances per category
    instances_per_category = {}
    for cat in cats:
        ann_ids = coco.getAnnIds(catIds=[cat['id']])
        instances_per_category[cat['name']] = len(ann_ids)
    
    # Count annotations per image
    annotations_per_image = {}
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations_per_image[len(ann_ids)] = annotations_per_image.get(len(ann_ids), 0) + 1
    
    # Count classes per image
    classes_per_image = {}
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        unique_classes = len(set(ann['category_id'] for ann in anns))
        classes_per_image[unique_classes] = classes_per_image.get(unique_classes, 0) + 1
    
    return {
        'n_images': n_images,
        'n_categories': n_categories,
        'n_annotations': n_annotations,
        'category_names': category_names,
        'instances_per_category': instances_per_category,
        'annotations_per_image': annotations_per_image,
        'classes_per_image': classes_per_image
    }

def plot_category_distributions(stats, output_dir='stats'):
    """Create visualizations of category distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data - sort by number of instances in descending order
    cat_counts = pd.Series(stats['instances_per_category']).sort_values(ascending=False)
    total_instances = cat_counts.sum()
    
    # 1. ALL categories horizontal bar plot with instance numbers
    plt.figure(figsize=(12, max(8, len(cat_counts) * 0.3)))  # Adjust height based on number of categories
    bars = plt.barh(range(len(cat_counts)), cat_counts.values)
    plt.yticks(range(len(cat_counts)), cat_counts.index)
    plt.xlabel('Number of Instances')
    plt.title('All Categories by Number of Instances')
    
    # Add instance numbers on the bars
    for i, (bar, count) in enumerate(zip(bars, cat_counts.values)):
        plt.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', fontsize=8)
    
    plt.gca().invert_yaxis()  # Most frequent at top
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_categories.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Distribution of images by number of classes per image
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(stats['classes_per_image'])
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Number of Classes per Image')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images by Number of Classes')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (x, count) in enumerate(zip(class_counts.index, class_counts.values)):
        plt.text(x, count + count * 0.01, f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classes_per_image.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print category statistics
    print("\nCategory Distribution Statistics:")
    print("-" * 50)
    print(f"Total number of categories: {len(cat_counts)}")
    print(f"Categories with >1000 instances: {sum(cat_counts > 1000)}")
    print(f"Categories with <100 instances: {sum(cat_counts < 100)}")
    print(f"\nTop 5 categories:")
    for cat, count in cat_counts.head().items():
        print(f"{cat}: {count:,} instances ({count/total_instances*100:.1f}%)")
    
    # Print classes per image statistics
    print(f"\nClasses per Image Statistics:")
    print("-" * 50)
    print(f"Average classes per image: {sum(class_counts.index * class_counts.values) / sum(class_counts.values):.2f}")
    print(f"Most common number of classes: {class_counts.idxmax()} ({class_counts.max():,} images)")
    print(f"Images with 1 class: {class_counts.get(1, 0):,}")
    print(f"Images with 2+ classes: {sum(class_counts[class_counts.index >= 2]):,}")

def plot_statistics(stats, output_dir='stats'):
    """Create visualizations of dataset statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style using seaborn
    sns.set_style("whitegrid")
    
    # Plot category distributions
    plot_category_distributions(stats, output_dir)
    
    # 2. Annotations per image distribution
    plt.figure(figsize=(10, 6))
    ann_counts = pd.Series(stats['annotations_per_image'])
    plt.bar(ann_counts.index, ann_counts.values)
    plt.title('Distribution of Annotations per Image')
    plt.xlabel('Number of Annotations')
    plt.ylabel('Number of Images')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annotations_per_image.png'))
    plt.close()
    
    # Print summary statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Number of Images: {stats['n_images']}")
    print(f"Number of Categories: {stats['n_categories']}")
    print(f"Total Annotations: {stats['n_annotations']}")
    print(f"Average Annotations per Image: {stats['n_annotations'] / stats['n_images']:.2f}")

def create_latex_table(stats, dataset_name="Ours"):
    """Create a LaTeX table with dataset statistics."""
    latex_str = """
\\begin{table}[h]
\\centering
\\begin{tabular}{|l|c|c|c|c|}
\\hline
Dataset & Categories & Instances & Images & Instances/Image \\\\
\\hline
%s & %d & %d & %d & %.1f \\\\
\\hline
\\end{tabular}
\\caption{Dataset Statistics}
\\label{tab:dataset_stats}
\\end{table}
""" % (
        dataset_name,
        stats['n_categories'],
        stats['n_annotations'],
        stats['n_images'],
        stats['n_annotations'] / stats['n_images']
    )
    
    with open(f'{dataset_name.lower()}_stats_table.tex', 'w') as f:
        f.write(latex_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO format dataset")
    parser.add_argument(
        "--data-dir", 
        default="data/20250507_NORD_FKB_Som_Korrigert",
        help="Dataset directory containing coco_dataset.json (default: data/20250507_NORD_FKB_Som_Korrigert)"
    )
    parser.add_argument(
        "--output-dir", 
        default="stats",
        help="Output directory for statistics and plots (default: stats)"
    )
    parser.add_argument(
        "--dataset-name", 
        default="Original",
        help="Name of the dataset for LaTeX table (default: Original)"
    )
    
    args = parser.parse_args()
    
    # Dataset path
    ann_file = os.path.join(args.data_dir, 'coco_dataset.json')
    
    if not os.path.exists(ann_file):
        print(f"Error: Annotation file not found: {ann_file}")
        exit(1)
    
    print(f"Analyzing dataset: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print("=" * 60)
    
    # Analyze dataset
    stats = analyze_dataset(ann_file)
    
    # Create visualizations
    plot_statistics(stats, args.output_dir)
    
    # Create LaTeX table
    create_latex_table(stats, args.dataset_name) 