#!/bin/bash

conda create -n shotluck python=3.10 -y
conda activate shotluck
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn==1.0.9 --no-build-isolation  # downgrade to flash attention v1 for older gpu support

echo "Setup script done! Make sure to upload data files to created directory"
