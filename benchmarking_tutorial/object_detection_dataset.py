import os
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms

class COCOObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, ann_file, transform=None):
        self.data_dir = data_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # Extract bounding boxes, labels, and segmentations
        bboxes = []
        labels = []
        segmentations = []
        for ann in anns:
            bbox = ann['bbox']
            # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bboxes.append(bbox)
            labels.append(ann['category_id'])
            
            # Load segmentation if available
            if 'segmentation' in ann and ann['segmentation'] is not None:
                segmentations.append(ann['segmentation'])
            else:
                segmentations.append(None)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, {'boxes': bboxes, 'labels': labels, 'segmentations': segmentations, 'image_id': img_id}

if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    data_dir = '20250507_NORD_FKB_Som_Korrigert'
    ann_file = os.path.join(data_dir, 'coco_dataset.json')
    dataset = COCOObjectDetectionDataset(data_dir, ann_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)