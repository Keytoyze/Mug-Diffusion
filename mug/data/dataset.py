import torch
import sqlite3

import audioread.ffdec
import librosa
import soundfile
import yaml
from pytorch_lightning import Callback
from torch.utils.data import Dataset
import cv2
import hashlib
from pytorch_lightning.utilities.distributed import rank_zero_only
# import sys
# import os
# sys.path.append(os.getcwd())

from mug import util
from mug.data.convertor import *


class OsuDataset(Dataset):
    def __init__(self,
                 txt_file,
                 feature_yaml=None,
                 sr=22050,
                 n_fft=2048,
                 max_audio_frame=16384,
                 audio_note_window_ratio=2,
                 n_mels=128,
                 mirror_p=0,
                 random_p=0,
                 shift_p=0,
                 rate_p=0,
                 feature_dropout_p=0,
                 mirror_at_interval_p=0,
                 freq_mask_p=0,
                 freq_mask_num=15,
                 rate=None,
                 test_txt_file=None,
                 with_audio=False,
                 with_feature=False,
                 cache_dir=None
                 ):
        self.data_paths = txt_file
        with open(self.data_paths, "r", encoding='utf-8') as f:
            self.beatmap_paths = f.read().splitlines()
        self.beatmap_paths = sorted(self.beatmap_paths, key=lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16))
        self.beatmap_paths = self.filter_beatmap_paths(self.beatmap_paths)

        self.feature_yaml = None
        self.with_feature = with_feature
        self.feature_dropout_p = feature_dropout_p
        if feature_yaml is not None and with_feature:
            self.feature_yaml = yaml.safe_load(open(feature_yaml))

        if test_txt_file is not None:
            with open(test_txt_file, "r", encoding='utf-8') as f:
                test_paths = f.read().splitlines()
                self.beatmap_paths = test_paths + self.beatmap_paths

        self.audio_hop_length = n_fft // 4
        self.audio_frame_duration = self.audio_hop_length / sr
        self.audio_note_window_ratio = audio_note_window_ratio
        self.convertor_params = {
            "frame_ms": self.audio_frame_duration * audio_note_window_ratio * 1000,
            "max_frame": max_audio_frame // audio_note_window_ratio
        }
        self.mirror_p = mirror_p
        self.random_p = random_p
        self.shift_p = shift_p
        self.rate_p = rate_p
        self.freq_mask_p = freq_mask_p
        self.freq_mask_num = freq_mask_num
        self.mirror_at_interval_p = mirror_at_interval_p
        self.with_audio = with_audio
        self.rate = rate
        self.sr = sr
        self.n_mels = n_mels
        self.max_audio_frame = max_audio_frame
        self.n_fft = n_fft
        self.max_duration = self.audio_frame_duration * max_audio_frame
        self.cache_dir = cache_dir
        self.error_files = []
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            # os.makedirs(os.path.join(cache_dir, 'processed'), exist_ok=True)
            error_path = os.path.join(self.cache_dir, "error.txt")
            if os.path.isfile(error_path):
                self.error_files = list(map(lambda x: x.strip(), open(error_path).readlines()))

    def __len__(self):
        return len(self.beatmap_paths)

    def load_audio_wave(self, audio_path, fallback_load_method=None):
        
        if len(fallback_load_method) == 0:
            raise ValueError(f"Cannot load: {audio_path}")
        # cache_name = (f"{os.path.basename(os.path.dirname(audio_path))}-"
        #               f"{os.path.basename(audio_path)}")
        # if not cache_name.endswith(".mp3"):
        #     cache_name += ".mp3"
        # cache_path = os.path.join(self.cache_dir, cache_name)
        # if not os.path.exists(cache_path):
        #     subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 
        #     'error', '-i', audio_path, '-ar', str(self.sr), cache_path])
        try:
            audio = fallback_load_method[0](audio_path)
            y, sr = librosa.load(audio, sr=self.sr, duration=self.max_duration)
            # assert sr == self.sr, f"sr = {sr}, but config sr = {self.sr}"
            if len(y) == 0:
                raise ValueError("")
            return y, sr
        except:
            return self.load_audio_wave(audio_path, fallback_load_method[1:])

    def load_audio_without_cache(self, audio_path, mel=True):
        # if '.mp3' in audio_path:
        #     os.system(f"ffmpeg -i '{audio_path}' -ar 22050 '{audio_path.replace('.mp3', 'sr.mp3')}'")
        #     audio_path = audio_path.replace('.mp3', 'sr.mp3')
        y, sr = self.load_audio_wave(audio_path, [soundfile.SoundFile,
                                                  audioread.ffdec.FFmpegAudioFile,
                                                  lambda x: x,
                                                #   soundfile.SoundFile
                                                  ])
        if mel:
            y = librosa.feature.melspectrogram(y=y, sr=sr,
                                               n_mels=self.n_mels,
                                               hop_length=self.audio_hop_length,
                                               n_fft=self.n_fft
                                               )
            y = np.log1p(y).astype(np.float16)
        else:
            # y_cuda = torch.FloatTensor(y).to('cuda:1')
            # st = time.time()
            # y2 = torch.stft(y_cuda, n_fft=self.n_fft, hop_length=self.audio_hop_length, return_complex=True).cpu().numpy()
            # print(time.time() - st, 'torch')
            # st = time.time()
            y = librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.audio_hop_length)
            # print(time.time() - st, 'librosa')
        return y

    def load_audio(self, audio_path):
        # cache_name = (f"{os.path.basename(os.path.dirname(audio_path))}-"
        #               f"{os.path.basename(audio_path)}.npz")
        # high_level_cache_path = os.path.join(self.cache_dir, "processed", cache_name)
        # if os.path.exists(high_level_cache_path):
        #     return np.load(high_level_cache_path)['y']
        # result = self.load_audio_without_cache(audio_path)
        # if abs(hash(cache_name)) % 5 == 1:
        #     np.savez_compressed(high_level_cache_path, y=result)
        # # print(audio_path, time.time() - st)
        # return result


        audio_path = audio_path.strip()
        if self.cache_dir is None:
            return self.load_audio_without_cache(audio_path)
        cache_name = (f"{os.path.basename(os.path.dirname(audio_path))}-"
                      f"{os.path.basename(audio_path)}.npz")
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.isfile(cache_path):
            return np.load(cache_path)['y']
        y = self.load_audio_without_cache(audio_path)
        np.savez_compressed(cache_path, y=y)
        return y

    def load_feature(self, path, dropout_prob=0, rate=1.0):
        name = os.path.basename(path)
        set_name = os.path.basename(os.path.dirname(path))
        feature_conn = sqlite3.Connection(os.path.join(
            os.path.dirname(self.data_paths),
            "feature.db"
        ))
        cursor = feature_conn.execute("SELECT * FROM Feature WHERE name = ? AND set_name = ?",
                                      [name, set_name])
        column_names = [description[0] for description in list(cursor.description)]
        result = cursor.fetchone()
        feature_dict = {}
        if result is not None:
            for i in range(len(column_names)):
                if random.random() >= dropout_prob:
                    feature_dict[column_names[i]] = result[i]
                    if column_names[i] == 'sr' and rate != 1.0:
                        if rate > 1:
                            star_ratio = 0.8184 * (rate - 1) + 1
                        else:
                            star_ratio = 1 / (0.8184 * (1 / rate - 1) + 1)
                        # print(f"change sr: {result[i]} -> {result[i] * star_ratio}, since rate change: x{rate}.")
                        feature_dict[column_names[i]] = result[i] * star_ratio
        emb_ids = util.feature_dict_to_embedding_ids(feature_dict, self.feature_yaml)
        # print(f"{path} -> {emb_ids}")
        return emb_ids

    def __getitem__(self, i):
        path = self.beatmap_paths[i]
        convertor_params = self.convertor_params.copy()
        convertor_params["mirror"] = np.random.random() < self.mirror_p
        convertor_params["random"] = np.random.random() < self.random_p
        convertor_params["mirror_at_interval_prob"] = self.mirror_at_interval_p
        convertor_params["offset_ms"] = 0
        convertor_params["rate"] = 1.0
        if self.rate is not None and np.random.random() < self.rate_p:
            # assert not self.with_audio, "Cannot change audio rate currently!"
            convertor_params["rate"] = np.random.random() * (self.rate[1] - self.rate[0]) + \
                                       self.rate[0]
        if np.random.random() < self.shift_p:
            assert not self.with_audio, "Cannot shift audio currently!"
            convertor_params["offset_ms"] = random.randint(0, int(convertor_params["max_frame"] *
                                                                  convertor_params["frame_ms"] / 2))
        try:
            objs, beatmap_meta = parse_osu_file(path, convertor_params)
            obj_array, valid_flag = beatmap_meta.convertor.objects_to_array(objs, beatmap_meta)
            example = {
                "meta": beatmap_meta.for_batch(),
                "convertor": convertor_params,
                "note": obj_array,
                "valid_flag": valid_flag
            }
            if self.with_audio:

                audio = self.load_audio(beatmap_meta.audio).astype(np.float32)

                if convertor_params["rate"] != 1.0:
                    t = int(round(audio.shape[1] / convertor_params["rate"]))
                    audio = cv2.resize(audio.reshape(self.n_mels, -1, 1), (t, self.n_mels))
                
                t = audio.shape[1]
                if t < self.max_audio_frame:
                    audio = np.concatenate([
                        audio,
                        np.zeros((self.n_mels, self.max_audio_frame - t), dtype=np.float32)
                    ], axis=1)
                elif t > self.max_audio_frame:
                    audio = audio[:, :self.max_audio_frame]
                
                if np.random.random() < self.freq_mask_p:
                    f = int(np.random.uniform(0, self.freq_mask_num)) # [0, F)
                    f0 = random.randint(0, self.n_mels - f) # [0, v - f)
                    audio[f0:f0 + f, :] = 0

                # audio = self.load_audio(beatmap_meta.audio)
                # if convertor_params["rate"] != 1.0:
                #     audio = librosa.phase_vocoder(audio, rate=convertor_params["rate"],
                #                                   hop_length=self.audio_hop_length)
                # t = audio.shape[1]
                # audio = np.concatenate([
                #     np.log1p(np.abs(audio)).reshape((1, -1, t)),
                #     np.angle(audio).reshape((1, -1, t))
                # ], axis=0)
                # if t < self.max_audio_frame:
                #     audio = np.concatenate([
                #         audio,
                #         np.zeros((2, self.n_fft // 2 + 1, self.max_audio_frame - t))
                #     ], axis=2)
                # elif t > self.max_audio_frame:
                #     audio = audio[:, :, :self.max_audio_frame]

                example["audio"] = audio.astype(np.float32)

                # debug_data = np.zeros((16 + self.n_mels, self.max_audio_frame))
                # for i in range(self.audio_note_window_ratio):
                #     debug_data[:16, np.arange(i, self.max_audio_frame, self.audio_note_window_ratio)] = obj_array * 5
                # debug_data[16:, :] = audio
                # breakpoint()

            if self.with_feature:
                example["feature"] = np.asarray(
                    self.load_feature(beatmap_meta.path, self.feature_dropout_p, convertor_params["rate"])
                )
            return example
        except Exception as e:
            if path not in self.error_files:
                with open(os.path.join(self.cache_dir, "error.txt"), "a+") as f:
                    f.write(f"{path}: {e}\n")
                self.error_files.append(path)
            # raise
            return self.__getitem__(random.randint(0, len(self.beatmap_paths) - 1))

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths


class OsuTrainDataset(OsuDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths[:int(len(beatmap_paths) * 0.8)]


class OsuValidDataset(OsuDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths[int(len(beatmap_paths) * 0.8):]


class BeatmapLogger(Callback):
    def __init__(self, log_batch_idx, count, splits=None, log_images_kwargs=None):
        super().__init__()
        self.log_batch_idx = log_batch_idx
        self.splits = splits
        self.count = count
        if log_images_kwargs is None:
            log_images_kwargs = {}
        self.log_images_kwargs = log_images_kwargs

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if batch_idx not in self.log_batch_idx:
            return
        if split not in self.splits:
            return
        if not hasattr(pl_module, "log_beatmap") or not callable(pl_module.log_beatmap):
            return
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        pl_module.log_beatmap(batch, split=split, count=self.count, **self.log_images_kwargs)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwarg):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                *args, **kwarg):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import yaml
    random.seed(0)
    dataset = OsuDataset(txt_file="data/beatmap_4k/beatmap.txt", n_fft=512, max_audio_frame=32768, audio_note_window_ratio=8, 
    n_mels=128, cache_dir="data/audio_cache", with_audio=True, with_feature=True, 
    feature_yaml="configs/mug/mania_beatmap_features.yaml"
    )
    dataset[0]


    
    # from tqdm import tqdm

    # os.makedirs(os.path.join("data", "audio_cache"), exist_ok=True)
    # base = (os.path.join("data", "audio_cache"))
    # for name in tqdm(os.listdir(base)):
    #     if name.endswith("npz"):
    #         y = np.load(os.path.join(base, name))['y']
    #         y = np.log1p(y).astype(np.float16)
    #         np.savez_compressed(os.path.join(os.path.join("data", "audio_cache_log_16"), name), y=y)
