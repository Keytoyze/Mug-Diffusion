![](asset/bg.jpg)

# MuG Diffusion

MuG Diffusion is a charting AI for rhythm games based on [Stable Diffusion](https://github.com/CompVis/latent-diffusion/) (one of the most powerful AIGC models) with a large modification to incorporate audio waves. Given an audio file, MuG Diffusion is able to generate high-quality diverse charts, which is aligned with the music and highly controllable. Currently, it supports 4K vertical scroll rhythm game (VSRG) only, with the following control options:

- Difficulty: supporting both [osu! star rating system](https://osu.ppy.sh/wiki/en/Beatmap/Star_rating) and [Etterna MSD system](https://etternaonline.com/).
- Style: ranked beatmaps (osu!) / stable charts (Malody), or other beatmap style. 
- LN: the ratio between long notes and regular notes.
- Patterns: supporting each pattern in Etterna MSD system, including chordjack, stamina, stream, jumpstream, handstream and technical.

MuG Diffusion aims to support other rhythm games in the future (osu!standard, 5-8K VSRG, maimai, etc), and hopes to provide a beneficial AIGC tool for all the charters and players.

![](asset/screenshot1.png)
![](asset/screenshot2.png)

## Installation and Running

### Bundled Executable

I packaged a bundled executable containing all the dependencies and model weights in the Windows platform, which is available [here](https://mug-diffusion-1305818561.cos.ap-nanjing.myqcloud.com/MugDiffusion.zip). Unzip the file and double click "Mug Diffusion.exe", which will open a browser interface for controlling. It takes around 30 seconds on my computer (NVidia 3050Ti, 4GB memory) to generate four charts for a 3-minute-long audio.


### Runing from Source

If you use other platforms or want to run from source, here is the instructions.

- Install Python.

- Install pytorch-cuda: https://pytorch.org/get-started/locally/

- Install other requirements:

```commandline
pip install -r requirements.txt
```

- Download the bundled executable, and copy the file `models/ckpt/model.ckpt` and `models/ckpt/model.yaml` to `{REPOSITORY_ROOT}/models/ckpt/*`.

- Run the WebUI:

```commandline
python webui.py
```

## Model Structure and Methodology

TODO

## Acknowledgement and Credits

Thank all the Charters / Mappers in the community. It's you who endowed MuG Diffusion with intelligence. Besides, I would like to thank the [Malody](https://m.mugzone.net/) development teams (and many other supporter) for the financial support.

Thank [zengrber](https://github.com/zengrber) for webui development, [RiceSS](https://osu.ppy.sh/users/8271436) for logo design, and many testers for their support. 

Charts created through MuG Diffusion are fully open source, explicitly falling under the CC0 1.0 Universal Public Domain Dedication.
