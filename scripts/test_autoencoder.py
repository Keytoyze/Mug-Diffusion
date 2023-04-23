import argparse, os, sys, glob
sys.path.append(".")

import numba
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
import yaml
from einops import rearrange
from torchvision.utils import make_grid
import eyed3

from mug.util import instantiate_from_config, feature_dict_to_embedding_ids, \
    load_audio_without_cache
from mug.diffusion.ddim import DDIMSampler
from mug.data.convertor import save_osu_file, BeatmapMeta, parse_osu_file
from mug.diffusion.diffusion import DDPM
import shutil


# from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template_beatmap",
        type=str,
        default="data/template.osu",
        help="path to a beatmap, serving as a template"
    )

    parser.add_argument(
        "--input_beatmap",
        type=str,
        help="path to an input beatmap"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/autoencoder"
    )


    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    opt = parser.parse_args()

    config = OmegaConf.load("models/ckpt/model.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model: DDPM = load_model_from_config(config, "models/ckpt/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    all_samples = list()
    with torch.no_grad():
        dataset = config.data.params.common_params

        # dataset.max_audio_frame = 16384
        # model.z_length = 256

        audio_hop_length = dataset.n_fft // 4
        audio_frame_duration = audio_hop_length / dataset.sr

        convertor_params = {
            "frame_ms": audio_frame_duration * dataset.audio_note_window_ratio * 1000,
            "max_frame": dataset.max_audio_frame // dataset.audio_note_window_ratio,
            "from_logits": True
        }

        beatmap_obj, beatmap_meta = parse_osu_file(opt.input_beatmap, convertor_params)
        obj_array, valid_flag = beatmap_meta.convertor.objects_to_array(beatmap_obj, beatmap_meta)
        obj_array_tensor = torch.FloatTensor(np.asarray([obj_array] * opt.n_samples)).to(device)
        z = model.model.first_stage_model.encode(obj_array_tensor)
        z = z.sample()
        out = model.model.first_stage_model.decode(z).detach().cpu().numpy()

        save_dir = os.path.join(opt.outdir)

        for i, x_sample in enumerate(out):
            convertor_params = convertor_params.copy()
            convertor_params["from_logits"] = True
            shutil.copyfile(beatmap_meta.audio, os.path.join(save_dir, os.path.basename(beatmap_meta.audio)))
            version = f"{beatmap_meta.version} - autoencoder [{i + 1}] [2]"
            creator = "MuG Diffusion"
            file_name = f"{version}.osu"

            save_osu_file(beatmap_meta, x_sample,
                          path=os.path.join(save_dir, file_name),
                          override={
                              "Version": version,
                              "AudioFilename": os.path.basename(beatmap_meta.audio),
                          })
        save_osu_file(beatmap_meta, obj_array,
                      path=os.path.join(save_dir, "raw.osu"),
                      override={
                          "Version": "raw",
                          "AudioFilename": os.path.basename(beatmap_meta.audio),
                      })

    print(f"Your samples are ready and waiting four you here: \n{save_dir} \nEnjoy.")
