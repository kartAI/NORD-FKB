import os
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
import numpy as np

class COCOSegmentationDataset(Dataset):
    def __init__(self, data_dir, ann_file, transform=None):
        self.data_dir = data_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        self.mask_dir = os.path.join(data_dir, 'Mask')
        
        # Create mapping from COCO category IDs to consecutive class indices
        self.cat_ids = sorted(self.coco.getCatIds())
        self.id_to_class = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Initialize a color mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # Iterate over each annotation to load and overlay the mask
        for ann_idx, ann in enumerate(anns):
            category_name = self.coco.loadCats(ann['category_id'])[0]['name']
            # Extract the base filename: remove Image_rgb/ prefix and .tif extension
            base_filename = img_info['file_name'].replace('Image_rgb/', '').replace('.tif', '')
            # Remove the Image_rgb part from the filename itself and fix double underscores
            base_filename = base_filename.replace('Image_rgb', '').replace('__', '_')
            mask_filename = f"{base_filename}-{ann_idx}-{category_name}.tif"
            mask_path = os.path.join(self.mask_dir, mask_filename)
            
            # Load mask
            ann_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if ann_mask is not None:
                # Map COCO category ID to consecutive class index
                class_idx = self.id_to_class[ann['category_id']]
                mask[ann_mask > 0] = class_idx

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()

        return image, mask




if __name__ == "__main__":
    # Define transformations
    # These are standard transformations for images (coming from imagenet)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    data_dir = '20250507_NORD_FKB_Som_Korrigert'
    ann_file = os.path.join(data_dir, 'coco_dataset.json')
    dataset = COCOSegmentationDataset(data_dir, ann_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)