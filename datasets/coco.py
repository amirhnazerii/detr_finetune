# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CITYSCAPES_CLASSES = {
    'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 
    'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7
}

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target




    
class CityscapesDetection(Dataset):
    def __init__(self, root_dir, ann_dir, transforms=None, return_masks=False, image_set='val'):
        self.root_dir = root_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.return_masks = return_masks
        self.image_set = image_set
        
        self.images = []
        self.targets = []
        
        # Find all city folders
        city_folders = [d for d in os.scandir(self.root_dir) if d.is_dir()]
        
        # Create COCO-compatible dataset structure
        self.coco_dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for name, id in CITYSCAPES_CLASSES.items():
            self.coco_dataset['categories'].append({
                'id': id,
                'name': name,
                'supercategory': 'object'
            })
        
        # Add images and annotations
        ann_id = 0
        for city_folder in city_folders:
            city_name = os.path.basename(city_folder.path)
            
            # Process images in this city
            for img_file in os.listdir(city_folder.path):
                if img_file.endswith('_leftImg8bit.png'):
                    # Get image path
                    img_path = os.path.join(city_folder.path, img_file)
                    
                    # Construct annotation path
                    base_name = img_file.replace('_leftImg8bit.png', '')
                    ann_file = f"{base_name}_gtFine_polygons.json"
                    ann_path = os.path.join(self.ann_dir, city_name, ann_file)
                    
                    if not os.path.exists(ann_path):
                        continue
                    
                    # Add image to list
                    self.images.append(img_path)
                    self.targets.append(ann_path)
                    
                    # Get image dimensions
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    # Create unique image id
                    img_id = len(self.coco_dataset['images'])
                    
                    # Add image info to COCO dataset
                    self.coco_dataset['images'].append({
                        'id': img_id,
                        'file_name': img_file,
                        'width': width,
                        'height': height
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
                            self.coco_dataset['annotations'].append({
                                'id': ann_id,
                                'image_id': img_id,
                                'category_id': CITYSCAPES_CLASSES[obj['label']],
                                'bbox': bbox,
                                'area': area,
                                'segmentation': segmentation,
                                'iscrowd': 0
                            })
                            
                            ann_id += 1
        
        # Create COCO api for evaluation
        if self.coco_dataset['annotations']:
            # Create a temporary file to load COCO
            coco_json_path = f'/tmp/cityscapes_{image_set}_coco.json'
            with open(coco_json_path, 'w') as f:
                json.dump(self.coco_dataset, f)
            
            self.coco = COCO(coco_json_path)
        else:
            self.coco = None
            print("WARNING: No annotations found in the dataset")
        
        print(f"Found {len(self.images)} image-annotation pairs in {image_set} set")
        print(f"Added {len(self.coco_dataset['annotations'])} annotations for {len(self.coco_dataset['images'])} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.targets[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Get image id in COCO
        img_id = idx  # We used the index as image_id
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and classes
        boxes = []
        classes = []
        
        for ann in coco_anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            classes.append(ann['category_id'])
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)
        
        # Create target compatible with DETR
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([img_id])
        
        # Add area and iscrowd for COCO evaluation
        if coco_anns:
            area = torch.tensor([ann['area'] for ann in coco_anns])
            iscrowd = torch.tensor([ann['iscrowd'] for ann in coco_anns])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Original image size
        target["orig_size"] = torch.as_tensor([h, w])
        target["size"] = torch.as_tensor([h, w])
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def get_height_and_width(self, idx):
        img_info = self.coco_dataset['images'][idx]
        return img_info['height'], img_info['width']
    

    
    
###############################################

    
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

    
    

    
def make_kitti_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),  # Flip with moderate probability
            T.RandomResize([352, 384, 416, 448], max_size=800),  # Match KITTI height
            normalize,
        ])

    elif image_set == 'val':
        return T.Compose([
            T.Resize((384, 1248)),  # Fixed resize matching KITTI shape while keeping ratio
            normalize,
        ])
    else:
        raise ValueError(f'Unknown image_set {image_set}')

    
    
    
    

def make_cityscapes_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1920),
                T.Compose([
                    T.RandomResize([800, 1000, 1200]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1920),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1920),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')    
    
    
    
    
    
###############################################    
    
    

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    
    if args.dataset_file == 'kitti':
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
        
    elif args.dataset_file == 'coco':
        dataset = CocoDetection(img_folder, ann_file, transforms=make_kitti_transforms(image_set), return_masks=args.masks)
    
    elif args.dataset_file == 'cityscapes':
        PATHS = {
            "train": (root / "leftImg8bit" / "train", root / "gtFine" / "train"),
            "val": (root / "leftImg8bit" / "val", root / "gtFine" / "val"),
        }
        img_folder, ann_folder = PATHS[image_set]
        dataset = CityscapesDetection(
            img_folder, 
            ann_folder, 
            transforms=make_cityscapes_transforms(image_set),
            return_masks=args.masks,
            image_set=image_set
        )

    else:
        raise ValueError(f'args.dataset_file: {args.dataset_file} is invalid.')
    return dataset
