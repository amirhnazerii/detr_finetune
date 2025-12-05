#!/bin/bash
# Submission helper: create and submit one SBATCH job per base model.
# Run this script from a login node (do NOT run it via `sbatch`).
# It will create per-model job scripts under ./sbatch_jobs/ and submit them with `sbatch`.

set -euo pipefail
IFS=$'\n\t'

#------------------------
# User-tweakable settings
#------------------------
WORKDIR="/home/anazeri/detr_finetune"
OUTPUT_PATH="/scratch/anazeri/detr_finetune/output"
CONDA_ENV="Detr_env1"
MODULE_CMD="module load"   # adjust to your cluster's module command if needed
MODULE_NAME="anaconda3/2023.09-0"
JOB_DIR="${WORKDIR}/sbatch_jobs"
CPUS_PER_TASK=2
MEM="100gb"
TIME="25:00:00"
GPUS=1                # used with --gres=gpu:${GPUS}
EPOCHS=100
NUM_CLASSES=9
DATASET="kitti"
COCO_PATH="/scratch/anazeri/kitti_coco_format/kitti_val/"
PYTHON_SCRIPT="main_modif_train.py"

#------------------------
# Models to submit
#------------------------
base_model_path_list=(
    "detr-r50-KITTI-orighead92fc.pth"
    "detr-r50dc5-KITTI-orighead92fc.pth"
    "detr-r101-KITTI-orighead92fc.pth"
    "detr-r101dc5-KITTI-orighead92fc.pth"
)

# Create job dir
mkdir -p "$JOB_DIR"

# Ensure sbatch exists
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch not found in PATH. Run this from a login node with Slurm's sbatch available."
    exit 2
fi

# Helper: set config per model
get_config() {
    local base_model_path="$1"
    case "$base_model_path" in
        *r50dc5*)
            backbone="resnet50"
            fintuned_model_path="${OUTPUT_PATH}/detr-r50dc5-KITTI-orighead92fc-${EPOCHS}epch"
            dilation_required="True"
            acronym="kt-r50dc5"
            ;;
        *r50*)
            backbone="resnet50"
            fintuned_model_path="${OUTPUT_PATH}/detr-r50-KITTI-orighead92fc-${EPOCHS}epch"
            dilation_required="False"
            acronym="kt-r50"
            ;;
        *r101dc5*)
            backbone="resnet101"
            fintuned_model_path="${OUTPUT_PATH}/detr-r101dc5-KITTI-orighead92fc-${EPOCHS}epch"
            dilation_required="True"
            acronym="kt-r101dc5"
            ;;
        *r101*)
            backbone="resnet101"
            fintuned_model_path="${OUTPUT_PATH}/detr-r101-KITTI-orighead92fc-${EPOCHS}epch"
            dilation_required="False"
            acronym="kt-r101"
            ;;
        *)
            echo "Unrecognized model pattern: $base_model_path"
            return 1
            ;;
    esac
}

# Submit one job per model and record IDs
declare -a SUBMITTED_JOBS=()

for base_model_path in "${base_model_path_list[@]}"; do
    get_config "$base_model_path"

    job_script_path="${JOB_DIR}/${acronym}_finetuned_${EPOCHS}.sh"
    log_out="${JOB_DIR}/${acronym}_finetuned_${EPOCHS}_%j.out"
    log_err="${JOB_DIR}/${acronym}_finetuned_${EPOCHS}_%j.err"

    cat > "$job_script_path" <<EOF
#!/bin/bash
#SBATCH --job-name=${acronym}_finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --gres=gpu:a100:${GPUS}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}

set -euo pipefail

cd "${WORKDIR}"
${MODULE_CMD} ${MODULE_NAME}
# activate conda env - adjust if your cluster uses a different activation method
source activate ${CONDA_ENV}

if [ "${GPUS}" -gt 1 ]; then
    echo "Launching with torchrun for ${GPUS} GPUs"
    torchrun --nproc_per_node=${GPUS} "${PYTHON_SCRIPT}" \
        --dataset_file "${DATASET}" \
        --coco_path "${COCO_PATH}" \
        --output_dir "${fintuned_model_path}" \
        --resume "${base_model_path}" \
        --num_classes ${NUM_CLASSES} \
        --backbone "${backbone}" \
        --dilation ${dilation_required} \
        --epochs ${EPOCHS}
else
    python "${PYTHON_SCRIPT}" \
        --dataset_file "${DATASET}" \
        --coco_path "${COCO_PATH}" \
        --output_dir "${fintuned_model_path}" \
        --resume "${base_model_path}" \
        --num_classes ${NUM_CLASSES} \
        --backbone "${backbone}" \
        --dilation ${dilation_required} \
        --epochs ${EPOCHS}
fi
EOF

    chmod +x "$job_script_path"
    echo "Submitting job script: $job_script_path"
    sbatch_out=$(sbatch "$job_script_path")
    # sbatch prints: Submitted batch job 123456
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    echo "Submitted ${acronym} as job ${job_id}"
    SUBMITTED_JOBS+=("${acronym}:${job_id}")
done

# Summary
printf '\nSubmitted jobs summary:\n'
for entry in "${SUBMITTED_JOBS[@]}"; do
    echo " - $entry"
done

echo "Job scripts are in: $JOB_DIR"

echo "Done. Monitor with: squeue -u \$USER | grep $(whoami) or sacct -j <jobid>"



