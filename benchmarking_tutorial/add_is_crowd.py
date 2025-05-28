import json

def add_fields_to_annotations(json_path):
    # Load the existing annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Add 'iscrowd' and 'area' fields to each annotation
    for ann in data['annotations']:
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
        
        if 'area' not in ann:
            # Calculate area from bbox
            x, y, width, height = ann['bbox']
            ann['area'] = width * height

    # Save the updated annotations
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# Path to your COCO dataset JSON file
json_path = '20250507_NORD_FKB_Som_Korrigert/coco_dataset.json'
add_fields_to_annotations(json_path)