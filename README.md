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
pip install flash-attn==1.0.9 --no-build-isolation  # downgrade to flash attention v1 for older GPUs
```

## Finetuning

Navigate to `model/scripts/tiny_llava/finetune/finetune.sh` and update the parameters for your image and annotation paths. For 1.5B LlaVa, change 3.1 to 1.5 and version to v1. Then, run `finetune.sh`

<!-- Model -->
## Model

Something here about model