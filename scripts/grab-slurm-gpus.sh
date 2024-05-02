#!/bin/bash
if [ "$1" = "1" ]; then
    echo "Requesting 8 GPUs for larger model"
    salloc --gres=gpu:H100:8 --time=0-02:00:00 --mem=128GB --ntasks-per-node=32
    nvidia-smi
    lscpu
elif [ "$1" = "0" ]; then
    pace-check-queue ice-gpu -s -c
else
    echo "Requesting 4 GPUs for smaller model"
    salloc --gres=gpu:H100:4 --time=0-04:00:00 --mem=64GB --ntasks-per-node=15
    nvidia-smi
    lscpu
fi
