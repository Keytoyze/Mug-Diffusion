import importlib
import math
from inspect import isfunction
import audioread.ffdec
import os
import numpy as np

import soundfile
import librosa

import torch


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def count_beatmap_features_embedding(x):
    if x['type'] == 'numeric':
        cur_count = int(math.ceil((x['max'] - x['min']) / x['interval'])) + 1
    elif x['type'] == 'category':
        cur_count = len(x['category']) + 1
    elif x['type'] == 'bool':
        cur_count = 3
    else:
        raise ValueError(str(x))
    return cur_count

def feature_dict_to_embedding_ids(feature_dict, feature_yaml):
    emb_ids = []
    current_emb_count = 0
    for x in feature_yaml:
        value = feature_dict.get(x['name'], None)
        if value is None:
            inter_index = 0  # missing
        else:
            if x['type'] == 'numeric':
                value = max(x['min'], min(x['max'], value))
                inter_index = int((value - x['min']) / x['interval'])
            elif x['type'] == 'bool':
                inter_index = value
            else:  # category
                try:
                    inter_index = x['category'].index(value)
                except IndexError:
                    inter_index = -1
            inter_index += 1  # 0 is missing
        for _ in range(x.get("count", 1)):
            emb_ids.append(inter_index + current_emb_count)
            current_emb_count += count_beatmap_features_embedding(x)
    return emb_ids

def count_beatmap_features(feature_yaml):
    count = 0
    for x in feature_yaml:
        count += count_beatmap_features_embedding(x) * x.get('count', 1)
    return count


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_dict_from_batch(dict_data, i):
    result = {}
    for k in dict_data:
        if isinstance(dict_data[k], torch.Tensor):
            result[k] = dict_data[k][i].item()
        else:
            result[k] = dict_data[k][i]
    return result

def load_audio_wave(sr, max_duration, audio_path, fallback_load_method=None):

    if len(fallback_load_method) == 0:
        raise ValueError(f"Cannot load: {audio_path}, {os.path.exists(audio_path)}")
    try:
        audio = fallback_load_method[0](audio_path)
        y, sr = librosa.load(audio, sr=sr, duration=max_duration)
        if len(y) == 0:
            raise ValueError("")
        return y, sr
    except:
        return load_audio_wave(sr, max_duration, audio_path, fallback_load_method[1:])

def load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration):
    y, sr = load_audio_wave(sr, max_duration, audio_path, [audioread.ffdec.FFmpegAudioFile,
                                                           soundfile.SoundFile, 
                                                           lambda x: x
                                                          ])
    y = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_mels=n_mels,
                                       hop_length=audio_hop_length,
                                       n_fft=n_fft
                                       )
    y = np.log1p(y).astype(np.float16)
    return y

def load_audio(cache_dir, audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration):
    audio_path = audio_path.strip()
    if cache_dir is None:
        return load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration)
    cache_name = (f"{os.path.basename(os.path.dirname(audio_path))}-"
                  f"{os.path.basename(audio_path)}.npz")
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.isfile(cache_path):
        return np.load(cache_path)['y']
    y = load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration)
    np.savez_compressed(cache_path, y=y)
    return y

if __name__ == '__main__':
    import yaml
    feature_yaml = yaml.safe_load(
        open("configs\mug\mania_beatmap_features.yaml")
    )
    print(feature_dict_to_embedding_ids(
        {"sr": 6.4, "ln_ratio": 0.0, "rc": True},
        feature_yaml
    ))
    print(feature_dict_to_embedding_ids(
        {"sr": 6.2, "ln_ratio": 0.5, "rc": False},
        feature_yaml
    ))
    print(feature_dict_to_embedding_ids(
        {"sr": 0, "ln_ratio": 0.5, "rc": True},
        feature_yaml
    ))
    print(feature_dict_to_embedding_ids(
        {"sr": 0.6, "hb": True},
        feature_yaml
    ))