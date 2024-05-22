#!/bin/bash
#SBATCH --job-name=eval_1.5_shot
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-7:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=16

module load anaconda3/2023.03
conda activate tinyllava

ROOT_DIR=$(pwd)

DATA_PATH="$ROOT_DIR/data/processed/annotations/20k_test_shots.json"
IMAGE_PATH="$ROOT_DIR/data/processed/videos"
OUTPUT_DIR="$ROOT_DIR/eval_data/3.1-caption"

MODEL_PATH="$ROOT_DIR/model_runs/ShotluckHolmes-3.1B"

python -m model.tinyllava.eval.model_shot2story \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --temperature 0.2 \
    --top_p 0.9 \
    --conv_mode v1 \
    --output_dir $OUTPUT_DIR \
    --mm_use_im_start_end False \
    --is_multimodal True

