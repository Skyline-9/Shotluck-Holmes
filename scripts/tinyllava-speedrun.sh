#!/bin/bash

git clone https://github.com/DLCV-BUAA/TinyLLaVABench.git
cd TinyLLaVABench
conda create -n tinyllava python=3.10 -y
conda activate tinyllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn==1.0.9 --no-build-isolation  # downgrade to flash attention v1 for older gpu support
mkdir data
current_dir=$(pwd)
sed -i 's|IMAGE_PATH=".*"|IMAGE_PATH="'"${current_dir}/data"'"|' scripts/tiny_llava/finetune/finetune.sh
sed -i 's|DATA_PATH=".*"|DATA_PATH="'"${current_dir}/data/test_finetune.json"'"|' scripts/tiny_llava/finetune/finetune.sh
sed -i 's|OUTPUT_DIR=".*"|OUTPUT_DIR="'test_output'"|' scripts/tiny_llava/finetune/finetune.sh

echo "Setup script done! Make sure to upload data files to created directory"
