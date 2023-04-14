![](asset/bg.jpg)

# MuG Diffusion

## Setup

- Install pytorch-cuda: https://pytorch.org/get-started/locally/

- Install other requirements:

```commandline
pip install -r requirements.txt
```

## Mapping

- Download ckpt file, and put it in `models/ckpt/model.ckpt` and `models/ckpt/model.yaml`.
- Start mapping by a gradio app:

```commandline
python webui.py
```