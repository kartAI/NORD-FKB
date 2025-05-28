# Data Checking Utilities

This folder contains Jupyter notebook code to check for missing files and masks in your dataset.

## How to Use

1. Place your dataset folder inside this directory. The dataset folder should contain the following:
   - `Mask/` (directory with mask files)
   - `Image_rgb/` (directory with image files)
   - `Metadata.json`
   - `coco_dataset.json`
2. Open and run the Jupyter notebook provided in this folder.
3. Specify the name of your dataset folder when prompted.
4. After running the notebook, an Excel file will be created that lists any missing data (images, masks, or metadata).

## Output
- The generated Excel file will summarize missing files and masks for easy review.

---

For questions or issues, please refer to the notebook or contact the repository maintainer.
