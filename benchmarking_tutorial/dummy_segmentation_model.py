import torch
import torch.nn as nn
import random
import numpy as np

class DummySegmentationModel(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DummySegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size

    def forward(self, images):
        # Generate random predictions
        batch_size = len(images)
        predictions = []
        for _ in range(batch_size):
            # Generate random shapes for each class
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.int64)
            
            # Generate 1-3 random shapes per class
            for class_id in range(1, self.num_classes + 1):
                num_shapes = random.randint(1, 3)
                for _ in range(num_shapes):
                    # Generate random circle or rectangle
                    shape_type = random.choice(['circle', 'rectangle'])
                    if shape_type == 'circle':
                        center_x = random.randint(0, self.image_size)
                        center_y = random.randint(0, self.image_size)
                        radius = random.randint(10, 50)
                        y, x = np.ogrid[:self.image_size, :self.image_size]
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        shape_mask = dist <= radius
                    else:  # rectangle
                        x1 = random.randint(0, self.image_size - 50)
                        y1 = random.randint(0, self.image_size - 50)
                        x2 = x1 + random.randint(20, 100)
                        y2 = y1 + random.randint(20, 100)
                        shape_mask = np.zeros((self.image_size, self.image_size), dtype=bool)
                        shape_mask[y1:y2, x1:x2] = True
                    
                    # Add the shape to the mask
                    mask[shape_mask] = class_id
            
            predictions.append(mask)
        return predictions