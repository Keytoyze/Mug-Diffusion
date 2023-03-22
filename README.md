# Mug-Diffusion

## Setup

- Install pytorch-cuda: https://pytorch.org/get-started/locally/

- Install other requirements:

```commandline
pip install -r requirements.txt
```

## Mapping

- Download ckpt file, and put it in `models/ckpt/model.ckpt` and `models/ckpt/model.yaml`.
- Modify prompt yaml files in `configs/mapping_config/feature_*.yaml`
- Start mapping:

```commandline
python scripts/mapping.py
    --audio path/to/audio/file
    --prompt_dir configs/mapping_config/
    --audio_title "Title"
    --audio_artist "Artist"
    --n_samples 4
```

Please make sure that `n_samples` should be the number of configuration files in `prompt_dir` which in the format of `feature_*.yaml`. For example, if `n_samples = 4`, then there should be feature_1.yaml, ..., feature_4.yaml in `prompt_dir`.