#!/bin/bash
#SBATCH --job-name=shot2story_eval1b5.sh
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-5:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=15

# print allocation
nvidia-smi

ROOT_DIR="/home/hice1/apeng39/scratch/Shotluck-Holmes/data"

DATA_PATH="$ROOT_DIR/converted_annotations/20k_test.json"
IMAGE_PATH="$ROOT_DIR/videos_extracted"
OUTPUT_DIR="$ROOT_DIR/OUTPUT"

# MODEL_PATH="/home/hice1/apeng39/scratch/tinyllava3.1b"
MODEL_PATH="/home/hice1/apeng39/scratch/tinyllava1.5b"

python -m tinyllava.eval.model_shot2story \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --temperature 0.2 \
    --conv_mode v1 \
    --output_dir $OUTPUT_DIR \
    --mm_use_im_start_end False \
    --is_multimodal True

