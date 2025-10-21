#!/usr/bin/env python3
"""
Simple training script for DINOv3-based segmentation model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from segmentation_dataset import COCOSegmentationDataset
from dinov3_segmentation_model import DINOv3SegmentationModel
from torchvision import transforms

def train_model():
    # Configuration
    train_data_dir = "data/NORD_FKB_train"
    train_ann_file = "data/NORD_FKB_train/coco_dataset.json"
    val_data_dir = "data/NORD_FKB_val"
    val_ann_file = "data/NORD_FKB_val/coco_dataset.json"
    
    epochs = 10
    batch_size = 2
    learning_rate = 0.001
    image_size = 512
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets first to get the correct number of classes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = COCOSegmentationDataset(train_data_dir, train_ann_file, transform=transform)
    val_dataset = COCOSegmentationDataset(val_data_dir, val_ann_file, transform=transform)
    
    # Get number of classes from dataset
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = DINOv3SegmentationModel(num_classes=num_classes, image_size=image_size)
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [target.to(device) for target in targets]  # Move targets to device
            
            optimizer.zero_grad()
            
            try:
                predictions = model(images)
                
                # Calculate loss
                loss = 0.0
                for pred, target in zip(predictions, targets):
                    # pred is [num_classes, height, width], target is [height, width]
                    # CrossEntropyLoss expects pred: [N, C, H, W] and target: [N, H, W]
                    loss += criterion(pred.unsqueeze(0), target.unsqueeze(0))
                
                loss = loss / len(predictions)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = [img.to(device) for img in images]
                targets = [target.to(device) for target in targets]  # Move targets to device
                
                try:
                    predictions = model(images)
                    
                    loss = 0.0
                    for pred, target in zip(predictions, targets):
                        # pred is [num_classes, height, width], target is [height, width]
                        # CrossEntropyLoss expects pred: [N, C, H, W] and target: [N, H, W]
                        loss += criterion(pred.unsqueeze(0), target.unsqueeze(0))
                    
                    loss = loss / len(predictions)
                    val_loss += loss.item()
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average validation loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'dinov3_segmentation_model.pth')
    print("Model saved!")

if __name__ == '__main__':
    train_model()
