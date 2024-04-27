#!/bin/bash

# Default values
gpu_count=2  # Default number of GPUs

# Check if an argument is provided for the number of GPUs
if [ "$#" -eq 1 ]; then
    gpu_count=$1
elif [ "$#" -eq 2 ]; then
    gpu_count=$1
    gpu_type=$2
fi

echo "Trying to grab $gpu_count $gpu_type"

salloc --gres=gpu:$gpu_type:$gpu_count --time=0-08:00:00 --mem=64GB --ntasks-per-node=15

nvidia-smi
