import argparse
import os
import sys

sys.path.append(".")

import torch
import numpy as np
from omegaconf import OmegaConf
import yaml
import eyed3

from mug.util import instantiate_from_config, feature_dict_to_embedding_ids, \
    load_audio_without_cache
from mug.diffusion.ddim import DDIMSampler
from mug.data.convertor import save_osu_file, parse_osu_file
from mug.diffusion.diffusion import DDPM
from mug.data.utils import gridify, remove_intractable_mania_mini_jacks
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


def parse_feature(batch_size, feature_dicts, feature_yaml, model: DDPM):
    features = []
    for i in range(batch_size):
        cur_dict = feature_dicts[i] if i < len(feature_dicts) else {}
        print(f"Feature {i}: {cur_dict}")
        features.append(feature_dict_to_embedding_ids(cur_dict, feature_yaml))
    feature = torch.tensor(np.asarray(features), dtype=torch.float32,
                           device=model.device)
    return model.model.cond_stage_model(feature)

def gridify_potassium(hit_objects):
    fraction = 4
    time_list = []
    for line in hit_objects:
        time = int(line.split(",")[2])
        time_list.append(time)
    if len(hit_objects) == 0:
        return
    start_time, end_time = time_list[0], time_list[-1]

    def filter_time(l):
        # <10ms 的 notes 视作一个，值取平均，保存数量
        epsilon = 10
        assert len(l) > 1

        l.append(2e9)
        result_idx = [0]
        for idx in range(1, len(l)):
            if l[idx] - l[result_idx[-1]] < epsilon:
                continue
            result_idx.append(idx)
        res = []
        for idx in range(0, len(result_idx) - 1):
            num = result_idx[idx + 1] - result_idx[idx]
            avg = sum(l[result_idx[idx]:result_idx[idx + 1]]) / num
            res.append((avg, num))
        return res

    # list of (avg_timestamp, num)
    der_list = filter_time(time_list)

    # 根据 offset 找 bpm，目的是让大部分都对齐在 1/12 线上
    def get_bpm(precision, offset):
        result_bpm = -1
        result_loss = 1e9
        result_result = {}
        for bpm in range(150 * precision, 300 * precision):
            bpm /= precision
            gap = 60 * 1000 / (fraction * bpm)
            loss = 0
            s, s2, notes = 0, 0, 0
            for (avg_time, cnt) in der_list:
                gap_time = avg_time - offset
                shang = round(gap_time / gap)
                delta = (gap_time - gap * shang)

                s += delta * cnt
                s2 += delta * delta * cnt
                notes += cnt
                # loss += delta * delta

            # sum delta^2/n
            loss = (s2 - 2 * s * (s / notes) + (s * s / notes / notes)) / notes

            loss /= gap

            if loss < result_loss:
                result_bpm = bpm
                result_loss = loss
            # if loss < 2:
            #     result_result[bpm] = loss
        print(result_loss)
        return result_bpm

    def get_offset(bpm, offset):
        # step 1:
        # 在这个 bpm 下，如何更改 offset 让尽可能多的音对在 1/1 1/2 1/4 线上？
        #          1/1    1/6 1/4 1/3     1/2    2/3 3/4 1/6
        if fraction == 12:
            weights = [100, 0, 20, 50, 60, 0, 100, 0, 60, 50, 20, 0]
        elif fraction == 4:
            weights = [100, 100, 100, 100]
        else:
            raise Exception("")
        gap = 60 * 1000 / (fraction * bpm)
        for precision_range in [range(-300, 300, 30), range(-30, 30, 5), range(-5, 5, 1)]:
            def get_val(offset):
                val = 0
                for (avg_time, cnt) in der_list:
                    gap_time = avg_time - offset
                    shang = round(gap_time / gap)
                    frac = shang % fraction
                    val += weights[frac] * cnt
                return val

            best_offset = offset
            best_val = get_val(offset)
            for i in precision_range:
                val = get_val(offset + i)
                if val > best_val:
                    best_val = val
                    best_offset = offset + i
            offset = best_offset

        # step 2: 试试能否做到让对音尽量准确
        s, tot = 0, 0
        for (avg_time, cnt) in der_list:
            gap_time = avg_time - offset
            shang = round(gap_time / gap)
            delta = (gap_time - gap * shang)
            s += delta * cnt
            tot += cnt
        offset += s / tot
        return offset

    offset = start_time
    for cur_precision in [10]:
        cur_bpm = get_bpm(cur_precision, offset)
        offset = get_offset(cur_bpm, offset)
        print(f"[{cur_precision}] bpm: {cur_bpm}, offset: {offset}")


    return cur_bpm, offset

# def gridify(hit_objects):
#     fraction = 4
#     time_list = []
#     for line in hit_objects:
#         time = int(line.split(",")[2])
#         time_list.append(time)
#     if len(hit_objects) == 0:
#         return
#     start_time, end_time = time_list[0], time_list[-1]
#
#     def filter_time(l):
#         # <10ms 的 notes 视作一个，值取平均，保存数量
#         epsilon = 10
#         assert len(l) > 1
#
#         l.append(2e9)
#         result_idx = [0]
#         for idx in range(1, len(l)):
#             if l[idx] - l[result_idx[-1]] < epsilon:
#                 continue
#             result_idx.append(idx)
#         res = []
#         for idx in range(0, len(result_idx) - 1):
#             num = result_idx[idx + 1] - result_idx[idx]
#             avg = sum(l[result_idx[idx]:result_idx[idx + 1]]) / num
#             res.append((avg, num))
#         return res
#
#     # list of (avg_timestamp, num)
#     der_list = filter_time(time_list)
#
#     # 根据 offset 找 bpm，目的是让大部分都对齐在 1/12 线上
#     def get_bpm(precision, offset):
#         result_bpm = -1
#         result_loss = 1e9
#         result_result = {}
#         for bpm in range(150 * precision, 300 * precision):
#             bpm /= precision
#             gap = 60 * 1000 / (fraction * bpm)
#             loss = 0
#             s, s2, notes = 0, 0, 0
#             for (avg_time, cnt) in der_list:
#                 gap_time = avg_time - offset
#                 shang = round(gap_time / gap)
#                 delta = (gap_time - gap * shang)
#
#                 s += delta * cnt
#                 s2 += delta * delta * cnt
#                 notes += cnt
#                 # loss += delta * delta
#
#             # sum delta^2/n
#             loss = (s2 - 2 * s * (s / notes) + (s * s / notes / notes)) / notes
#
#             loss /= gap
#
#             if loss < result_loss:
#                 result_bpm = bpm
#                 result_loss = loss
#             # if loss < 2:
#             #     result_result[bpm] = loss
#         print(result_loss)
#         return result_bpm
#
#     def get_offset(bpm, offset):
#         # step 1:
#         # 在这个 bpm 下，如何更改 offset 让尽可能多的音对在 1/1 1/2 1/4 线上？
#         #          1/1    1/6 1/4 1/3     1/2    2/3 3/4 1/6
#         if fraction == 12:
#             weights = [100, 0, 20, 50, 60, 0, 100, 0, 60, 50, 20, 0]
#         elif fraction == 4:
#             weights = [100, 100, 100, 100]
#         else:
#             raise Exception("")
#         gap = 60 * 1000 / (fraction * bpm)
#         for precision_range in [range(-300, 300, 30), range(-30, 30, 5), range(-5, 5, 1)]:
#             def get_val(offset):
#                 val = 0
#                 for (avg_time, cnt) in der_list:
#                     gap_time = avg_time - offset
#                     shang = round(gap_time / gap)
#                     frac = shang % fraction
#                     val += weights[frac] * cnt
#                 return val
#
#             best_offset = offset
#             best_val = get_val(offset)
#             for i in precision_range:
#                 val = get_val(offset + i)
#                 if val > best_val:
#                     best_val = val
#                     best_offset = offset + i
#             offset = best_offset
#
#         # step 2: 试试能否做到让对音尽量准确
#         s, tot = 0, 0
#         for (avg_time, cnt) in der_list:
#             gap_time = avg_time - offset
#             shang = round(gap_time / gap)
#             delta = (gap_time - gap * shang)
#             s += delta * cnt
#             tot += cnt
#         offset += s / tot
#         return offset
#
#     offset = start_time
#     for cur_precision in [10]:
#         cur_bpm = get_bpm(cur_precision, offset)
#         offset = get_offset(cur_bpm, offset)
#         print(f"[{cur_precision}] bpm: {cur_bpm}, offset: {offset}")
#     cur_gap = 60 * 1000 / cur_bpm
#
#     def attaching(t_start):
#         result = str(t_start)
#         t = t_start - offset
#         base_offset = int(t / cur_gap) * cur_gap
#         note_offset = t - base_offset
#         for div in [1, 2, 3, 4, 6, 8]:
#             gap_div = cur_gap / div
#             gap_grid_time = round(note_offset / gap_div) * gap_div
#             gap_offset = abs(note_offset - gap_grid_time)
#             if gap_offset <= 10:
#                 result = str(int(gap_grid_time + base_offset + offset))
#                 break
#             # if div == 8:
#             #     result = str(int(gap_grid_time + base_offset + offset))
#         return result
#
#     gridified_hit_objects = []
#     for line in hit_objects:
#         elements = line.split(",")
#         elements[2] = attaching(int(elements[2]))
#         if int(elements[3]) == 128:
#             e = elements[5].split(":")
#             e[0] = attaching(int(e[0]))
#             elements[5] = ":".join(e)
#         gridified_hit_objects.append(",".join(elements))
#
#     return cur_bpm, offset, gridified_hit_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="configs/mapping_config",
        help="the prompt to render, in yaml config"
    )

    parser.add_argument(
        "--feature_yaml",
        type=str,
        default="configs/mug/mania_beatmap_features.yaml",
        help="the prompt to render, in yaml config"
    )

    parser.add_argument(
        "--template_beatmap",
        type=str,
        default="data/template.osu",
        help="path to a beatmap, serving as a template"
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="path to audio file"
    )

    parser.add_argument(
        "--audio_title",
        type=str,
        default=None,
        help="title of the audio"
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="bpm of the audio"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="offset of the audio"
    )

    parser.add_argument(
        "--audio_artist",
        type=str,
        default=None,
        help="artist of the audio"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/beatmaps"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--no_adsorption",
        action='store_true',
        help="don't adsorpt the notes to grids",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    audio_file = eyed3.load(opt.audio)
    audio_title = audio_file.tag.title if opt.audio_title is None else opt.audio_title
    audio_artist = audio_file.tag.artist if opt.audio_artist is None else opt.audio_artist

    config = OmegaConf.load("models/ckpt/model.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ckpt/model.ckpt")  # TODO: check path

    if opt.prompt_dir is not None:
        feature_dicts = []
        for i in range(opt.n_samples):
            feature_dicts.append(yaml.safe_load(open(os.path.join(opt.prompt_dir, f"feature_{i + 1}.yaml"))))
    else:
        feature_dicts = []

    feature_yaml = yaml.safe_load(open(opt.feature_yaml))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        assert False
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    all_samples = list()
    with torch.no_grad():
        uc = None
        if opt.scale != 1.0:
            uc = parse_feature(opt.n_samples, {}, feature_yaml, model)

        c = parse_feature(opt.n_samples, feature_dicts, feature_yaml, model)
        dataset = config.data.params.common_params

        # dataset.max_audio_frame = 16384
        # model.z_length = 256

        audio_hop_length = dataset.n_fft // 4
        audio_frame_duration = audio_hop_length / dataset.sr

        audio = load_audio_without_cache(opt.audio, dataset.n_mels, dataset.n_fft // 4,
                                     dataset.n_fft, dataset.sr,
                                     audio_frame_duration * dataset.max_audio_frame)
        t = audio.shape[1]
        if t < dataset.max_audio_frame:
            audio = np.concatenate([
                audio,
                np.zeros((dataset.n_mels, dataset.max_audio_frame - t), dtype=np.float32)
            ], axis=1)
        elif t > dataset.max_audio_frame:
            audio = audio[:, :dataset.max_audio_frame]
        # print(opt.n_samples)
        w = torch.tensor(
            np.stack([audio for _ in range(opt.n_samples)]),
                       dtype=torch.float32).to(model.device)
        w = model.model.wave_model(w)
        # shape = [4, opt.H//8, opt.W//8]
        shape = None
        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                         c=c, w=w,
                                         batch_size=opt.n_samples,
                                         shape=shape,
                                         verbose=True,
                                         unconditional_guidance_scale=opt.scale,
                                         unconditional_conditioning=uc,
                                         eta=opt.ddim_eta)

        x_samples_ddim = model.model.decode(samples_ddim).cpu().numpy()

        template_path = opt.template_beatmap
        template_name = os.path.basename(template_path)
        save_dir = os.path.join(opt.outdir, f"{audio_artist} - {audio_title}")
        os.makedirs(save_dir, exist_ok=True)
        convertor_params = {
            "frame_ms": audio_frame_duration * dataset.audio_note_window_ratio * 1000,
            "max_frame": dataset.max_audio_frame // dataset.audio_note_window_ratio
        }

        def custom_gridify(hit_objects):
            hit_objects = remove_intractable_mania_mini_jacks(hit_objects)
            hit_objects, bpm, offset = gridify(hit_objects)
            return bpm, offset, hit_objects

        for i, x_sample in enumerate(x_samples_ddim):
            convertor_params = convertor_params.copy()
            convertor_params["from_logits"] = True
            _, beatmap_meta = parse_osu_file(template_path, convertor_params)
            shutil.copyfile(opt.audio, os.path.join(save_dir, "audio.mp3"))
            version = f"AI v{i + 1} ({config.version}) - refined"
            creator = "MuG Diffusion"
            file_name = f"{audio_artist} - {audio_title} ({creator}) [{version}].osu".replace("/", "")

            save_osu_file(beatmap_meta, x_sample,
                          path=os.path.join(save_dir, file_name),
                          override={
                              "Creator": f"{creator} v{config.version}",
                              "Version": version,
                              "AudioFilename": "audio.mp3",
                              "Title": audio_title,
                              "TitleUnicode": audio_title,
                              "Artist": audio_artist,
                              "ArtistUnicode": audio_artist,
                          }, gridify=custom_gridify)

    print(f"Your samples are ready and waiting four you here: \n{save_dir} \nEnjoy.")
