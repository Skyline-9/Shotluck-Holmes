#!/bin/bash

git clone https://github.com/DLCV-BUAA/TinyLLaVABench.git
cd TinyLLaVABench
conda create -n tinyllava python=3.10 -y
conda activate tinyllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
mkdir data
sed -i 's|IMAGE_PATH=".*"|IMAGE_PATH="'/content/TinyLLaVABench/data'"|' scripts/tiny_llava/finetune/finetune.sh
sed -i 's|DATA_PATH=".*"|DATA_PATH="'/content/TinyLLaVABench/data/test_finetune.json'"|' scripts/tiny_llava/finetune/finetune.sh

echo "Setup script done! Make sure to upload data files to created directory"
