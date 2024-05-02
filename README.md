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
    <li>
        <a href="#data-pre-processing">Data Pre-processing</a>
        <ul>
            <li><a href="#downloading">Downloading</a></li>
            <li><a href="#pre-processing">Pre-processing</a></li>
        </ul>
    </li>
    <li><a href="#finetuning">Finetuning</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->

## Introduction

Video is a rapidly growing format rich in information, yet it remains a challenging task for computers to understand. A
video often consists of a storyline comprised of multiple short shots, and comprehension of the video requires not only
understanding the shot-by-shot visual-audio information but also associating the ideas between each shot for a larger
big-picture idea. Despite significant progress, current works neglect videos' more granular shot-by-shot semantic
information. In this project, we propose an efficient large language vision model (LLVM) to boost video summarization
and captioning. Specifically, we reproduce near-SOTA results on the Shot2Story video captioning task with a much smaller
model.

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
cd model
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
cd ..
pip install flash-attn==2.5.8 --no-build-isolation  # upgrade to this version of flash-attn for H100
# pip install flash-attn==1.0.9 --no-build-isolation  # downgrade to flash attention v1 for older GPUs
```

Alternatively, you can run `setup-speedrun.sh` from the root directory to execute all the commands above

```shell
sh scripts/setup-speedrun.sh
```

## Data Pre-processing

***Note: all the following commands should be run from the project root directory***

### Downloading

Raw annotations should already be downloaded with this repository. If your annotations are missing, download the
annotations by running

```shell
sh data/scripts/download/download_annotations.sh
```

If running on Shot2Story dataset, follow https://github.com/bytedance/Shot2Story/issues/5 to download the data
and extract the videos into `data/raw/videos`.

### Pre-processing

First, process the videos by running `process_videos.py` in scripts/data/process, which will run `ffmpeg` to split
the shot videos into different files. Then, convert the annotation data and scan for corrupted videos by
running `convert_shot2story_to_llava.py`

Set processes to a reasonable number depending on how many CPU cores you have available.

```sh
python scripts/data/process/process_videos.py --processes=<YOUR_NUM_PROCESSES>
python scripts/data/process/convert_shot2story_to_llava.py
```

If you plan on running eval, make sure to run `convert_shot2story_to_llava.py` on the test set as well.

_Note_: `ffmpeg` is required for process_videos.py. If this is not installed, download ffmpeg accordingly for your OS or
install it locally using the `download-ffmpeg.sh` script.

## Finetuning

Finetuning scripts are in `scripts/run/finetune`. Run the finetuning script corresponding to which model you want to
use.

```sh
sh scripts/run/finetune/finetune_1b5.sh  # finetune the 1.5B model
```

```sh
sh scripts/run/finetune/finetune_3b1.sh  # finetune the 3.1B model
```

<!-- Model -->

## Model

Full model metrics, model zoo, and more details coming soon!
