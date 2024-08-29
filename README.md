<!-- PROJECT LOGO -->
<br />
<p align="center">
<!--   <a href="https://github.com/Skyline-9/Shotluck-Holmes">
    <img src="logo.jpeg" alt="Logo" width="140" height="120" >
  </a> -->

  <h1 align="center">🔍 Shotluck Holmes</h1>

  <p align="center">
    Large Language Vision Models For Shot-Level Video Understanding (Richard Luo, Austin Peng, Adithya Vasudev, Rishabh Jain)
    <br /><br />
    <a href="https://arxiv.org/abs/2405.20648"><strong>Read the Preprint Here »</strong></a>
    <br /><br />
    <img src="https://img.shields.io/github/license/Skyline-9/Shotluck-Holmes?style=for-the-badge" alt="GitHub License">
    <a href="https://paperswithcode.com/sota/video-captioning-on-shot2story20k?p=shotluck-holmes-a-family-of-efficient-small">
      <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shotluck-holmes-a-family-of-efficient-small/video-captioning-on-shot2story20k&style=for-the-badge" alt="Shotluck Holmes Badge">
    </a>
    <a href="https://paperswithcode.com/sota/video-summarization-on-shot2story20k?p=shotluck-holmes-a-family-of-efficient-small">
      <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shotluck-holmes-a-family-of-efficient-small/video-summarization-on-shot2story20k&style=for-the-badge" alt="Shotluck Holmes Badge">
    </a>
  </p>
</p>

<div align="center">
</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#-requirements-and-installation">🔧 Requirements and Installation</a></li>
    <li>
        <a href="#data-pre-processing">Data Pre-processing</a>
        <ul>
            <li><a href="#downloading">Downloading</a></li>
            <li><a href="#pre-processing">Pre-processing</a></li>
        </ul>
    </li>
    <li><a href="#finetuning">Finetuning</a></li>
    <li><a href="#model">Model</a></li>
    <li><a href="#results">Results</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->

## Introduction

  Video is an increasingly prominent and information-dense medium, yet it poses substantial challenges for language models. A typical video consists of a sequence of shorter segments, or shots, that collectively form a coherent narrative. Each shot is analogous to a word in a sentence where multiple data streams of information (such as visual and auditory data) must be processed simultaneously. Comprehension of the entire video requires not only understanding the visual-audio information of each shot but also requires that the model links the ideas between each shot to generate a larger, all-encompassing story. Despite significant progress in the field, current works often overlook videos’ more granular shot-by-shot semantic information. In this project, we propose a family of efficient large language vision models (LLVMs) to boost video summarization and captioning called Shotluck Holmes. By leveraging better pretraining and data collection strategies, we extend the abilities of existing small LLVMs from being able to understand a picture to being able to understand a sequence of frames. Specifically, we show that Shotluck Holmes achieves better performance than state-of-the-art results on the Shot2Story video captioning and summary task with significantly smaller and more computationally efficient models.

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

Hugging Face Models
- [Shotluck Holmes 1.5B Model](https://huggingface.co/RichardLuo/Shotluck-Holmes-1.5)
- [Shotluck Holmes 3.1B Model](https://huggingface.co/RichardLuo/Shotluck-Holmes-3.1)

## Results

*Table 1: Performance of best models on single-shot video captioning*

| Model                     | BLEU     | METEOR   | ROUGE   | CIDER     |
|---------------------------|----------|----------|---------|-----------|
| Shot2Story (7B+)          | **10.7** | 16.2     | 29.6    | 37.4      |
| Shotluck-Holmes (3.1B)    | 8.7      | **25.7** | 36.2    | 63.2      |
| Shotluck-Holmes (1.5B)    | 9.3      | 25.3     | **36.3**| **68.9**  |

*Table 2: Performance of best models on multi-shot video summarization*

| Model                     | BLEU  | METEOR | ROUGE | CIDER  |
|---------------------------|-------|--------|-------|--------|
| Shot2Story (7B+)          | **11.7** | 19.7   | 26.8  | 8.6    |
| Shotluck-Holmes (3.1B)    | 7.67  | **23.2** | **43**  | **152.3** |
| Shotluck-Holmes (1.5B)    | 6.48  | 21.3   | 40.2  | 144.3  |

