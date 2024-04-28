#!/bin/bash
if [ "$1" = "1" ]; then
    echo "Requesting 4 GPUs for larger model"
    salloc --gres=gpu:4 --time=0-04:00:00 --mem=64GB --ntasks-per-node=15
    nvidia-smi
    lscpu
elif [ "$1" = "0" ]; then
    pace-check-queue ice-gpu -s -c
else
    echo "Requesting 2 GPUs for smaller model"
    salloc --gres=gpu:2 --time=0-08:00:00 --mem=32GB --ntasks-per-node=15
    nvidia-smi
    lscpu
fi