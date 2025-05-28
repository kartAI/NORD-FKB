import torch
import torch.nn as nn
import random

class DummyObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DummyObjectDetectionModel, self).__init__()
        self.num_classes = num_classes

    def forward(self, images):
        # Generate random predictions
        batch_size = len(images)
        predictions = []
        for _ in range(batch_size):
            num_boxes = random.randint(1, 5)  # Random number of boxes per image
            boxes = torch.rand((num_boxes, 4)) * 512  # Random boxes within image size
            labels = torch.randint(1, self.num_classes, (num_boxes,))  # Random labels
            scores = torch.rand((num_boxes,))  # Random confidence scores
            predictions.append({'boxes': boxes, 'labels': labels, 'scores': scores})
        return predictions