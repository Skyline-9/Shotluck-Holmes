#!/bin/bash
#SBATCH --job-name=finetune_1b5.sh
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=15

# print allocation
nvidia-smi

DATA_PATH="/home/hice1/avasudev8/scratch/nlp/Shotluck-Holmes/data/my_annotations/20k_test.json"
IMAGE_PATH="/home/hice1/avasudev8/scratch/nlp/Shotluck-Holmes/data/videos_extracted"
OUTPUT_DIR="/home/hice1/avasudev8/scratch/nlp/Shotluck-Holmes/data/OUTPUT"

MODEL_PATH="/home/hice1/avasudev8/scratch/nlp/Shotluck-Holmes/data/tinyllava1.5b"


python -m tinyllava.eval.model_shot2story \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --temperature 0.2 \
    --conv_mode llava_v0 \
    --output_dir $OUTPUT_DIR \
    --mm_use_im_start_end False \
    --is_multimodal True 

