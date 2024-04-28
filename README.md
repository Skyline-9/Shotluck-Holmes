<!-- PROJECT LOGO -->
<br />
<p align="center">
<!--   <a href="https://github.com/Skyline-9/Shotluck-Holmes">
    <img src="logo.jpeg" alt="Logo" width="140" height="120" >
  </a> -->

  <h1 align="center">Shotluck Holmes</h1>

  <p align="center">
    Large Language Vision Models For Shot-Level Video Understanding (Richard Luo, Austin Peng, Adithya Vasudev, Rishabh Jain)
    <br />
    <br />
<!--     <a href="https://arxiv.org/pdf/2005.09007.pdf"><strong>Read the paper »</strong></a> -->
    <img src="https://img.shields.io/github/license/Skyline-9/Shotluck-Holmes?style=for-the-badge" alt="GitHub License">
    <br />
    <br />
  </p>
</p>

<div align="center">
</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li>
      <a href="#model">Model</a>
    </li>
    <li><a href="#-requirements-and-installation">🔧 Requirements and Installation</a></li>
    <li><a href="#finetuning">Finetuning</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->
## Introduction

Something about task motivation

<!-- REQUIREMENTS AND INSTALLATION -->
## 🔧 Requirements and Installation

1. Clone this repository and navigate to the folder

```sh
git clone https://github.com/Skyline-9/Shotluck-Holmes.git
cd Shotluck-Holmes
```

2. Install packages

```sh
conda create -n shotluck python=3.10 -y
conda activate shotluck
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation  # upgrade to this version of flash-attn for H100
# pip install flash-attn==1.0.9 --no-build-isolation  # downgrade to flash attention v1 for older GPUs
```

## Data Pre-processing

If running on Shot2Story dataset, follow https://github.com/bytedance/Shot2Story/issues/5 to download the data and then run `process_videos.py` in data/scripts. Then, convert the data by running the following from the root directory

```sh
python data/scripts/convert_shot2story_to_llava.py --p YOUR_INPUT_PATH --o YOUR_OUTPUT_FILE
```

## Finetuning

Finetuning scripts are in `model/scripts/tiny_llava/finetune`. First, edit the data and image path in the scripts

```
DATA_PATH="YOUR_FULL_PATH_TO_ANNOTATIONS/test.json"
IMAGE_PATH="YOUR_FULL_PATH_TO_VIDEOS"
OUTPUT_DIR="OUTPUT_FOLDER_NAME"
```

Run the finetuning script corresponding to which model you want to use. For example, for the 1.5B model, run
```sh
sh model/scripts/tiny_llava/finetune/finetune_1b5.sh
```

Similarly, the 3.1B model is in `finetune_3b1.sh`.

<!-- Model -->
## Model

Something here about model
