#!/bin/bash

#SBATCH --job-name=fine_detr_Xep
#SBATCH --nodes=1                    # Keep it on one node to avoid inter-node communication overhead
#SBATCH --tasks-per-node=2            # Use 4 processes per node
#SBATCH --cpus-per-task=4             # Allocate 8 CPU cores per process
#SBATCH --mem=150gb                   # Set 100GB memory (adjust if needed)
#SBATCH --time=15:00:00               # 12 hours; adjust based on expected training time
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=finetune_detr_newhead_batchjob_v2.out
#SBATCH --error=finetune_detr_newhead_batchjob_v2.err



cd /home/anazeri/detr_finetune/


module add anaconda3/2023.09-0
source activate Detr_env1

# srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main_finetune_modifhead.py \
#     --dataset_file "coco" \
#     --coco_path "/scratch/anazeri/coco/" \
#     --output_dir "/home/anazeri/detr_finetune/detr-r50-coco-modifhead-128fc92fc-epoch10" \
#     --resume "detr-r50-modifhead-128fc92fc.pth" \
#     --lr_drop 3 \
#     --backbone "resnet50" \
#     --epochs 5


python main_finetune_modifhead.py   --dataset_file "coco"   --coco_path "/scratch/anazeri/coco/"   --output_dir '/home/anazeri/detr_finetune/detr-r50-coco-modifhead-128fc92fc-TEMP'   --resume 'detr-r50-modifhead-128fc92fc.pth'   --lr_drop 7 --backbone 'resnet50' --new_layer_dim 128  --epochs 10
