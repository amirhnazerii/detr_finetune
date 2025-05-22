# python cityscapes_to_coco.py --cityscapes-dir "/scratch/anazeri/cityscapes" --output-dir "/scratch/anazeri/cityscapes/annotations" --split "train"
# python cityscapes_to_coco.py --cityscapes-dir "/scratch/anazeri/cityscapes" --output-dir "/scratch/anazeri/cityscapes/annotations" --split "val"



#!/bin/bash
# convert_cityscapes.sh

# Set paths
CITYSCAPES_DIR="/scratch/anazeri/cityscapes"
COCO_OUTPUT_DIR="/scratch/anazeri/cityscapes_coco"

# Create output directory
mkdir -p $COCO_OUTPUT_DIR

# Convert train split
echo "Converting train split..."
python cityscapes_to_coco.py \
    --cityscapes-dir $CITYSCAPES_DIR \
    --output-dir $COCO_OUTPUT_DIR \
    --split train

# Convert val split
echo "Converting val split..."
python cityscapes_to_coco.py \
    --cityscapes-dir $CITYSCAPES_DIR \
    --output-dir $COCO_OUTPUT_DIR \
    --split val

echo "Conversion complete! COCO-format dataset is in $COCO_OUTPUT_DIR"