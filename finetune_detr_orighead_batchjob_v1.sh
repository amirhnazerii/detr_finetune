#!/bin/bash

#SBATCH --job-name=a-city_train_v4_detr_70epch
#SBATCH --nodes=1                    # Keep it on one node to avoid inter-node communication overhead
#SBATCH --tasks-per-node=2            # Use 4 processes per node
#SBATCH --cpus-per-task=2             # Allocate 8 CPU cores per process
#SBATCH --mem=100gb                   # Set 100GB memory (adjust if needed)
#SBATCH --time=12:00:00               # 12 hours; adjust based on expected training time
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=cityscapes_detr_orighead_finetune_v4_epch70_batchjob.out
#SBATCH --error=cityscapes_detr_orighead_finetune_v4_epch70_batchjob.err


cd /home/anazeri/detr_finetune/

module add anaconda3/2023.09-0
source activate Detr_env1


# python main_finetune_modifhead.py   --dataset_file "coco"   --coco_path "/scratch/anazeri/coco/"   --output_dir '/home/anazeri/detr_finetune/robust-detr-r50-coco-modifhead-128fc92fc-TEMP'   --resume 'detr-r50-modifhead-128fc92fc.pth'   --lr_drop 7 --backbone 'resnet50' --new_layer_dim 128 --inter_class_weight 0 --robust True  --epochs 10
#python main_modif_train.py   --dataset_file "coco"   --coco_path "/scratch/anazeri/kitti_coco_format/kitti_val/"   --output_dir '/home/anazeri/detr_finetune/detr-r50-KITTI-orighead92fc-40epch'   --resume 'detr-r50-KITTI-orighead92fc.pth' --num_classes 9 --backbone 'resnet50'  --epochs 40

# ### Fine-grained:
# python main_modif_train.py   --dataset_file "kitti"   --coco_path "/scratch/anazeri/kitti_coco_format/kitti_val/"   --output_dir '/home/anazeri/detr_finetune/detr-r50-KITTI-orighead92fc-50epch'   --resume 'detr-r50-KITTI-orighead92fc.pth' --num_classes 9 --backbone 'resnet50'  --epochs 50


# ### Fine-tuned:
# python main_modif_train.py   \
#     --dataset_file "cityscapes" \
#     --coco_path "/scratch/anazeri/cityscapes_coco" \
#     --output_dir '/home/anazeri/detr_finetune/detr-orighead-r50-cityscapes-finetune-v4-70epch' \
#     --resume 'detr-r50-cityscapes-orighead.pth' \
#     --num_classes 8 \
#     --backbone 'resnet50' \
#     --epochs 70


dataset="kitti"
dataset_path="/scratch/anazeri/kitti_coco_format/kitti_val/"
fintuned_model_path='/home/anazeri/detr_finetune/detr-r50dc5-KITTI-orighead92fc-50epch'
base_model_path='detr-r50-cityscapes-orighead.pth'


if 'r50' in $fintuned_model_path:
    backbone='resnet50'
elif 'r50dc5' in $fintuned_model_path:
    backbone='resnet50'
elif 'r101' in $fintuned_model_path:

elif 'r101dc5' in $fintuned_model_path:

### Fine-tuned:
python main_modif_train.py   \
    --dataset_file $dataset \
    --coco_path $dataset_path \
    --output_dir $fintuned_model_path \
    --resume $base_model_path \
    --num_classes 9 \
    --backbone $backbone \
    --epochs 70



