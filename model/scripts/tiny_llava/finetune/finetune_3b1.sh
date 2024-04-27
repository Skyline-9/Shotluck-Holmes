#!/bin/bash

# Assign the arguments to variables
DATA_PATH="/home/hice1/apeng39/scratch/Shotluck-Holmes/data/my_annotations/20k_val.json"
IMAGE_PATH="/home/hice1/apeng39/scratch/Shotluck-Holmes/data/videos"
OUTPUT_DIR="/home/hice1/apeng39/scratch/Shotluck-Holmes/data/OUTPUT"

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path bczhou/TinyLLaVA-3.1B \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower bczhou/TinyLLaVA-3.1B-SigLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
