# Adding Segmentation Masks to the COCO Dataset

## Purpose of `add_segmentation.py`

The `add_segmentation.py` script is a utility designed to augment your COCO-format dataset with segmentation mask information. While the COCO format supports segmentation masks, your original dataset may only contain bounding boxes or lack the required segmentation fields. This script fills that gap by reading individual mask files and encoding them into the COCO JSON annotations.

## Why Do You Need to Run This Script?

For benchmarking segmentation models, the COCO evaluation tools require each annotation in your dataset to include a `segmentation` field (in RLE format) and an `area` field. If your dataset was originally created for object detection or lacks these fields, you must add them before running any segmentation benchmarks.

Currently, this process is not automated in the data preparation pipeline, so you need to run `add_segmentation.py` **once** to update your dataset. This ensures that all ground truth annotations have the necessary segmentation information for evaluation.

## How Do We Add Segmentation Data?

The segmentation data is added by associating each annotation in the COCO JSON file with a corresponding mask file stored in the `Mask/` directory. Each mask file is a grayscale image (typically `.tif` format) where the object of interest is marked with nonzero pixel values (usually 255 for the object, 0 for the background).

The script:
- Matches each annotation to its mask file using a naming convention: `{image_name}-{annotation_index}-{category_name}.tif`.
- Loads the mask image as a binary mask (object pixels > 0).
- Encodes the binary mask using COCO's Run-Length Encoding (RLE) format.
- Adds the RLE-encoded mask to the annotation's `segmentation` field.
- Calculates the area (number of object pixels) and adds it to the annotation's `area` field.

## What is the Mask Encoded As?

COCO expects the `segmentation` field to be in RLE (Run-Length Encoding) format for efficiency. Here's how it works:

- **Binary Mask:** The mask image is first converted to a binary mask (object = 1, background = 0).
- **RLE Encoding:** The binary mask is then encoded using the COCO API's RLE format, which compresses the mask for storage and evaluation.
- **JSON Storage:** The RLE-encoded mask is stored as a dictionary in the `segmentation` field of each annotation in the JSON file. It contains two keys:
  - `counts`: The RLE-encoded string
  - `size`: The height and width of the mask

Example:
```json
"segmentation": {
    "counts": "eWc11...",
    "size": [512, 512]
}
```

## How Does the Script Work?

- **Inputs:**
  - The path to your COCO dataset JSON file (e.g., `coco_dataset.json`)
  - The path to your data directory (containing the `Mask/` folder with individual mask files)
- **Process:**
  1. Loads the COCO JSON annotations.
  2. For each annotation, finds the corresponding mask file in the `Mask/` directory.
  3. Reads the mask, encodes it in COCO's RLE format, and adds it to the annotation.
  4. Computes the area of the mask and adds it to the annotation.
  5. Saves a new JSON file (e.g., `coco_dataset_with_segmentation.json`) with the updated annotations.

## When Should You Run It?

- **Run once** after preparing your dataset and before running segmentation benchmarks.
- If you update your masks or annotations, re-run the script to keep the segmentation information in sync.

## Example Usage

```bash
python add_segmentation.py
```

By default, the script uses hardcoded paths. You may edit the script to point to your dataset if needed.

## Future Improvements

In the future, this step may be integrated into the data preparation pipeline, making it unnecessary to run manually. For now, running this script ensures your dataset is ready for segmentation benchmarking. 