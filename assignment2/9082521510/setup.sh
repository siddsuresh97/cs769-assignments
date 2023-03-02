#!/usr/bin/env bash

conda create -n bert_hw python=3.7
conda activate bert_hw


# removing this line due to errors in creating 1.8 pytorch environment in multiples gpus i tried
# conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch

# errors that i faced:
# /python3.7/site-packages/torch/cuda/__init__.py:104: UserWarning:
# NVIDIA A100-PCIE-40GB with CUDA capability sm_80 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
# If you want to use the NVIDIA A100-PCIE-40GB GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

# NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.


# to word around this and also make sure to have pytorch 1.8.0, i have used the folowing command:
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
