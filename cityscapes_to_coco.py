# #!/usr/bin/env python3
# # cityscapes_to_coco.py

# import os
# import json
# import argparse
# from pathlib import Path
# from PIL import Image
# import numpy as np

# # Define the Cityscapes classes for instance segmentation
# CITYSCAPES_CLASSES = {
#     'person': 1, 
#     'rider': 2, 
#     'car': 3, 
#     'truck': 4, 
#     'bus': 5, 
#     'train': 6, 
#     'motorcycle': 7, 
#     'bicycle': 8
# }

# def convert_cityscapes_to_coco(cityscapes_dir, output_file, split='val'):
#     """
#     Convert Cityscapes dataset to COCO format
#     Args:
#         cityscapes_dir: Path to the Cityscapes dataset
#         output_file: Path to save the COCO format JSON file
#         split: Dataset split ('train', 'val')
#     """
#     images_dir = os.path.join(cityscapes_dir, 'leftImg8bit', split)
#     annots_dir = os.path.join(cityscapes_dir, 'gtFine', split)
    
#     # Create COCO dataset structure
#     coco_dataset = {
#         'images': [],
#         'annotations': [],
#         'categories': []
#     }
    
#     # Add categories (starting from 1 to match COCO convention)
#     for name, id in CITYSCAPES_CLASSES.items():
#         coco_dataset['categories'].append({
#             'id': id,
#             'name': name,
#             'supercategory': 'object'
#         })
    
#     # Get city folders
#     city_folders = [d for d in os.scandir(images_dir) if d.is_dir()]
    
#     # Process images and annotations
#     ann_id = 1  # Start annotation IDs at 1 (COCO convention)
#     print(f"Processing {split} split of Cityscapes dataset...")
    
#     for city_folder in city_folders:
#         city_name = os.path.basename(city_folder.path)
#         print(f"Processing city: {city_name}")
        
#         # Process images in this city
#         for img_file in os.listdir(city_folder.path):
#             if img_file.endswith('_leftImg8bit.png'):
#                 # Get image path
#                 img_path = os.path.join(city_folder.path, img_file)
                
#                 # Construct annotation path
#                 base_name = img_file.replace('_leftImg8bit.png', '')
#                 ann_file = f"{base_name}_gtFine_polygons.json"
#                 ann_path = os.path.join(annots_dir, city_name, ann_file)
                
#                 if not os.path.exists(ann_path):
#                     print(f"Warning: Annotation not found for {img_file}")
#                     continue
                
#                 # Get image dimensions
#                 img = Image.open(img_path)
#                 width, height = img.size
                
#                 # Create unique image id (must be an integer for COCO API)
#                 # Use a hash of the file path to create a unique integer
#                 img_id = hash(img_path) % 100000
                
#                 # Add image info to COCO dataset
#                 coco_dataset['images'].append({
#                     'id': img_id,
#                     'file_name': f"{city_name}/{img_file}",  # Store path relative to leftImg8bit
#                     'width': width,
#                     'height': height,
#                     'license': 1,
#                     'flickr_url': '',
#                     'coco_url': '',
#                     'date_captured': ''
#                 })
                
#                 # Load and process annotations
#                 with open(ann_path, 'r') as f:
#                     anno_data = json.load(f)
                
#                 for obj in anno_data['objects']:
#                     if obj['label'] in CITYSCAPES_CLASSES:
#                         # Only keep instances defined in our class mapping
#                         polygon = np.array(obj['polygon'])
                        
#                         # Skip invalid polygons
#                         if len(polygon) < 3:
#                             continue
                        
#                         # Get bounding box from polygon
#                         x_min, y_min = np.min(polygon, axis=0)
#                         x_max, y_max = np.max(polygon, axis=0)
                        
#                         # Skip tiny objects
#                         if (x_max - x_min < 10) or (y_max - y_min < 10):
#                             continue
                        
#                         # Convert to COCO format [x, y, width, height]
#                         bbox = [float(x_min), float(y_min), 
#                                float(x_max - x_min), float(y_max - y_min)]
                        
#                         # Calculate area
#                         area = float(bbox[2] * bbox[3])
                        
#                         # Convert polygon to COCO segmentation format
#                         segmentation = [[float(p[0]), float(p[1])] for p in polygon]
#                         segmentation = [sum(segmentation, [])]  # Flatten list
                        
#                         # Add annotation
#                         coco_dataset['annotations'].append({
#                             'id': ann_id,
#                             'image_id': img_id,
#                             'category_id': CITYSCAPES_CLASSES[obj['label']],
#                             'bbox': bbox,
#                             'area': area,
#                             'segmentation': segmentation,
#                             'iscrowd': 0
#                         })
                        
#                         ann_id += 1
    
#     # Save the COCO dataset
#     print(f"Saving COCO format annotations to {output_file}")
#     print(f"Dataset contains {len(coco_dataset['images'])} images and {len(coco_dataset['annotations'])} annotations")
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'w') as f:
#         json.dump(coco_dataset, f)
    
#     print("Conversion complete!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Convert Cityscapes dataset to COCO format')
#     parser.add_argument('--cityscapes-dir', required=True, help='Path to Cityscapes dataset')
#     parser.add_argument('--output-dir', required=True, help='Path to save COCO format annotations')
#     parser.add_argument('--split', default='val', choices=['train', 'val'], help='Dataset split')
    
#     args = parser.parse_args()
    
#     output_file = os.path.join(args.output_dir, f"instances_{args.split}.json")
#     convert_cityscapes_to_coco(args.cityscapes_dir, output_file, args.split)


#!/usr/bin/env python3
# cityscapes_to_coco.py

import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Define the Cityscapes classes for instance segmentation
CITYSCAPES_CLASSES = {
    'person': 1, 
    'rider': 2, 
    'car': 3, 
    'truck': 4, 
    'bus': 5, 
    'train': 6, 
    'motorcycle': 7, 
    'bicycle': 8
}

def convert_cityscapes_to_coco(cityscapes_dir, output_dir, split='val', copy_images=True):
    """
    Convert Cityscapes dataset to COCO format
    Args:
        cityscapes_dir: Path to the Cityscapes dataset
        output_dir: Path to save COCO format dataset
        split: Dataset split ('train', 'val')
        copy_images: Whether to copy images to a COCO-style structure
    """
    # Setup paths
    images_dir = os.path.join(cityscapes_dir, 'leftImg8bit', split)
    annots_dir = os.path.join(cityscapes_dir, 'gtFine', split)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    annotations_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Target image directory
    target_img_dir = os.path.join(output_dir, f'{split}2017')  # COCO-style naming
    if copy_images:
        os.makedirs(target_img_dir, exist_ok=True)
    
    # Create COCO dataset structure
    coco_dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories (starting from 1 to match COCO convention)
    for name, id in CITYSCAPES_CLASSES.items():
        coco_dataset['categories'].append({
            'id': id,
            'name': name,
            'supercategory': 'object'
        })
    
    # Get city folders
    city_folders = [d for d in os.scandir(images_dir) if d.is_dir()]
    
    # Process images and annotations
    ann_id = 1  # Start annotation IDs at 1 (COCO convention)
    img_id = 1  # Start image IDs at 1
    
    print(f"Processing {split} split of Cityscapes dataset...")
    
    for city_folder in city_folders:
        city_name = os.path.basename(city_folder.path)
        print(f"Processing city: {city_name}")
        
        # Process images in this city
        for img_file in os.listdir(city_folder.path):
            if img_file.endswith('_leftImg8bit.png'):
                # Get image path
                img_path = os.path.join(city_folder.path, img_file)
                
                # Construct annotation path
                base_name = img_file.replace('_leftImg8bit.png', '')
                ann_file = f"{base_name}_gtFine_polygons.json"
                ann_path = os.path.join(annots_dir, city_name, ann_file)
                
                if not os.path.exists(ann_path):
                    print(f"Warning: Annotation not found for {img_file}")
                    continue
                
                # Get image dimensions
                img = Image.open(img_path)
                width, height = img.size
                
                # Create COCO-style filename (unique ID with .jpg extension for compatibility)
                # COCO uses 12-digit zero-padded image IDs
                coco_filename = f"{img_id:012d}.png"
                
                # Copy the image if requested
                if copy_images:
                    target_img_path = os.path.join(target_img_dir, coco_filename)
                    shutil.copy2(img_path, target_img_path)
                
                # Add image info to COCO dataset
                coco_dataset['images'].append({
                    'id': img_id,
                    'file_name': coco_filename,
                    'width': width,
                    'height': height,
                    'license': 1,
                    'flickr_url': '',
                    'coco_url': '',
                    'date_captured': ''
                })
                
                # Load and process annotations
                with open(ann_path, 'r') as f:
                    anno_data = json.load(f)
                
                for obj in anno_data['objects']:
                    if obj['label'] in CITYSCAPES_CLASSES:
                        # Only keep instances defined in our class mapping
                        polygon = np.array(obj['polygon'])
                        
                        # Skip invalid polygons
                        if len(polygon) < 3:
                            continue
                        
                        # Get bounding box from polygon
                        x_min, y_min = np.min(polygon, axis=0)
                        x_max, y_max = np.max(polygon, axis=0)
                        
                        # Skip tiny objects
                        if (x_max - x_min < 10) or (y_max - y_min < 10):
                            continue
                        
                        # Convert to COCO format [x, y, width, height]
                        bbox = [float(x_min), float(y_min), 
                               float(x_max - x_min), float(y_max - y_min)]
                        
                        # Calculate area
                        area = float(bbox[2] * bbox[3])
                        
                        # Convert polygon to COCO segmentation format
                        segmentation = [[float(p[0]), float(p[1])] for p in polygon]
                        segmentation = [sum(segmentation, [])]  # Flatten list
                        
                        # Add annotation
                        coco_dataset['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': CITYSCAPES_CLASSES[obj['label']],
                            'bbox': bbox,
                            'area': area,
                            'segmentation': segmentation,
                            'iscrowd': 0
                        })
                        
                        ann_id += 1
                
                # Increment image ID
                img_id += 1
    
    # Save the COCO annotations
    output_file = os.path.join(annotations_dir, f'instances_{split}2017.json')
    print(f"Saving COCO format annotations to {output_file}")
    print(f"Dataset contains {len(coco_dataset['images'])} images and {len(coco_dataset['annotations'])} annotations")
    
    with open(output_file, 'w') as f:
        json.dump(coco_dataset, f)
    
    print("Conversion complete!")
    if copy_images:
        print(f"Images have been copied to {target_img_dir}")
    
    return {
        'annotations_file': output_file,
        'images_dir': target_img_dir if copy_images else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Cityscapes dataset to COCO format')
    parser.add_argument('--cityscapes-dir', required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--output-dir', required=True, help='Path to save COCO format dataset')
    parser.add_argument('--split', default='val', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--no-copy-images', action='store_true', help='Do not copy images (only create annotations)')
    
    args = parser.parse_args()
    
    convert_cityscapes_to_coco(
        args.cityscapes_dir, 
        args.output_dir, 
        args.split, 
        not args.no_copy_images
    )