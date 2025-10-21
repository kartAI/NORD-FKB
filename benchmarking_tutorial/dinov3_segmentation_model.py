import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
from PIL import Image

class DINOv3SegmentationModel(nn.Module):
    def __init__(self, num_classes, image_size=512):
        super(DINOv3SegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Use DINOv2 feature extractor (DINOv3 not available in transformers)
        self.feature_extractor = pipeline(
            model="facebook/dinov2-base",
            task="image-feature-extraction",
        )
        
        # Simple segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Store device for later use
        self.device = None
    
    def forward(self, images):
        """Forward pass"""
        predictions = []
        
        # Get device from first image
        if self.device is None:
            self.device = next(self.parameters()).device
        
        for image in images:
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                image = Image.fromarray(img_np)
            
            # Extract features
            features = self.feature_extractor(image)
            features = torch.tensor(features, device=self.device)  # [1, 257, 768] or [257, 768] or [768]
            
            # Handle different feature shapes
            if features.dim() == 3:
                # Features are [1, num_patches, 768] - remove batch dim and take mean
                features = features.squeeze(0)  # [257, 768]
                features = features.mean(dim=0)  # [768]
            elif features.dim() == 2:
                # Features are [num_patches, 768] - take mean
                features = features.mean(dim=0)  # [768]
            elif features.dim() == 1:
                # Features are already [768]
                pass
            
            # Reshape to spatial format for conv2d: [batch, channels, height, width]
            features = features.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 768, 1, 1]
            
            # Segmentation head
            segmentation = self.segmentation_head(features)
            segmentation = F.interpolate(segmentation, size=(self.image_size, self.image_size), mode='bilinear')
            
            # Return logits (not argmax) for loss calculation
            segmentation = segmentation.squeeze(0)  # Remove batch dimension
            predictions.append(segmentation)
        
        return predictions
