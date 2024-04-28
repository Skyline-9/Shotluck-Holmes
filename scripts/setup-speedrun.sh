#!/bin/bash

conda create -n shotluck python=3.10 -y
conda activate shotluck
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation  # required for H100 support

echo "Setup script done! Make sure to upload data files to created directory"
