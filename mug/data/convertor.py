import os
import random
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class BeatmapMeta:
    path: str
    audio: str = ""
    game_mode: int = 0
    convertor: 'BaseOsuConvertor' = None
    cs: float = 0
    version: str = ""
    file_meta: List[str] = field(default_factory=lambda: [])

    def for_batch(self):
        result = asdict(self, dict_factory=lambda x: {k: v
                                                      for (k, v) in x
                                                      if k != 'convertor' and k != 'file_meta'})
        return result


def read_item(line):
    return line.split(":")[-1].strip()


def parse_osu_file(osu_path, convertor_params: Optional[dict]) -> Tuple[List[str], BeatmapMeta]:
    data = open(osu_path, 'r', encoding='utf-8').read().split("\n")
    parsing_context = ""
    hit_objects = []
    meta = BeatmapMeta(path=osu_path)
    for line in data:
        line = line.strip()

        if parsing_context == "[HitObjects]" and "," in line:
            hit_objects.append(line)
        else:
            if line != "[HitObjects]":
                meta.file_meta.append(line)

            if parsing_context == "[General]":
                if line.startswith("AudioFilename"):
                    meta.audio = os.path.join(os.path.dirname(osu_path),
                                              read_item(line))
                elif line.startswith("Mode"):
                    meta.game_mode = int(read_item(line))
                    if convertor_params is not None:
                        meta.convertor = MOD_CONVERTOR[meta.game_mode](**convertor_params)

            elif parsing_context == "[Metadata]":
                if line.startswith("Version"):
                    meta.version = read_item(line)

            elif parsing_context == "[Difficulty]":
                if line.startswith("CircleSize"):
                    meta.cs = float(read_item(line))

        if line.startswith("["):
            parsing_context = line

    return hit_objects, meta


def save_osu_file(meta: BeatmapMeta, note_array: np.ndarray, path=None, override=None):
    convertor = meta.convertor
    hit_objects = convertor.array_to_objects(note_array, meta)
    with open(path, "w", encoding='utf8') as f:
        for line in meta.file_meta:
            if override is not None:
                for k, v in override.items():
                    if line.startswith(k):
                        line = f"{k}: {v}"
                        break
            f.write(line + "\n")

        f.write("[HitObjects]\n")
        for hit_object in hit_objects:
            f.write(hit_object + "\n")


class BaseOsuConvertor(metaclass=ABCMeta):

    def __init__(self, frame_ms, max_frame, mirror=False, from_logits=False, offset_ms=0,
                 random=False, rate=1.0):
        self.frame_ms = frame_ms
        self.max_frame = max_frame
        self.mirror = mirror
        self.from_logits = from_logits
        self.offset_ms = offset_ms
        self.random = random
        self.rate = rate


    @abstractmethod
    def objects_to_array(self, hit_objects: List[str],
                         meta: BeatmapMeta) -> Tuple[np.ndarray, np.ndarray]: pass

    @abstractmethod
    def array_to_objects(self, note_array: np.ndarray, meta: BeatmapMeta) -> List[str]: pass


class OsuManiaConvertor(BaseOsuConvertor):
    def is_binary_positive(self, input):
        if self.from_logits:
            return input > 0
        else:
            return input > 0.5

    """
    Feature Layout:
        [is_start: 0/1] * key_count

        [offset_start: 0-1] * key_count
        valid only if is_start = 1

        [is_holding: 0/1] * key_count, (exclude start, include end),
        valid only if previous.is_start = 1 or previous.is_holding = 1

        [offset_end: 0-1]
        valid only if is_holding = 1 and latter.is_holding = 0
    """

    def read_time(self, text):
        t = int(text) / self.rate + self.offset_ms
        index = int(t / self.frame_ms)
        offset = (t - index * self.frame_ms) / self.frame_ms
        return int(round(t)), index, offset

    def array_to_objects(self, note_array: np.ndarray, meta: BeatmapMeta) -> List[str]:
        note_array = note_array.transpose()
        hit_object_with_start = []
        key_count = int(meta.cs)
        column_width = int(512 / key_count)
        for column in range(key_count):
            start_indices = np.where(self.is_binary_positive(note_array[:, column]))[0]
            for start_index in start_indices:
                start_offset = np.clip(note_array[start_index, column + key_count], 0, 1)
                start = int(round((start_index + start_offset) * self.frame_ms))
                end = -1

                if start_index != len(note_array) - 1:
                    i = start_index + 1
                    while (i < len(note_array)
                           and self.is_binary_positive(note_array[i, column + key_count * 2])
                           and not self.is_binary_positive(note_array[i, column])):
                        i += 1
                    end_index = i - 1
                    if end_index == start_index:
                        end = -1
                    else:
                        end_offset = np.clip(note_array[end_index, column + key_count * 3], 0, 1)
                        end = int(round((end_index + end_offset) * self.frame_ms))

                column_num = int(round((column + 0.5) * column_width))
                if end == -1:
                    line = f"{column_num},192,{start},1,0,0:0:0:0:"
                else:
                    line = f"{column_num},192,{start},128,0,{end}:0:0:0:0:"
                hit_object_with_start.append((line, start))
        hit_object_with_start = sorted(hit_object_with_start, key=lambda x: x[1])
        return list(map(lambda x: x[0], hit_object_with_start))

    def objects_to_array(self, hit_objects: List[str],
                         meta: BeatmapMeta) -> Tuple[np.ndarray, np.ndarray]:
        key_count = int(meta.cs)
        column_width = int(512 / key_count)
        array = np.zeros((self.max_frame, key_count * 4),
                         dtype=np.float32)
        max_index = 0

        column_map = list(range(key_count))
        if self.mirror:
            column_map = [key_count - column_map[i] - 1 for i in range(key_count)]
        if self.random:
            random.shuffle(column_map)

        for line in hit_objects:
            params = line.split(",")
            column = int(int(float(params[0])) / column_width)
            if column >= key_count or column < 0:
                continue
            column = column_map[column]

            start, start_index, start_offset = self.read_time(params[2])
            # is_start / offset_start
            if start_index >= len(array):
                continue
            array[start_index, column] = 1
            array[start_index, column + key_count] = start_offset
            max_index = max(start_index, max_index)

            # LN
            if int(params[3]) == 128:
                end, end_index, end_offset = self.read_time(params[5].split(":")[0])
                if end_index >= len(array):
                    end_index = len(array) - 1
                    end_offset = 1
                for i in range(start_index + 1, end_index + 1):
                    array[i, column + key_count * 2] = 1
                array[end_index, column + key_count * 3] = end_offset
                max_index = max(end_index, max_index)

        valid_flag = np.zeros((len(array),))
        valid_flag[:max_index] = 1
        array = np.transpose(array)
        return array, valid_flag


MOD_CONVERTOR = {
    3: OsuManiaConvertor
}

if __name__ == "__main__":
    map_path = """E:\E\osu!\Songs\891164 Various Artists - 4K LN Dan Courses v2 - Extra Level -\Various Artists - 4K LN Dan Courses v2 - Extra Level - (_underjoy) [13th Dan - Yoru (Marathon)].osu"""
    objs, beatmap_meta = parse_osu_file(map_path, convertor_params={"frame_ms": 2048 / 22050 / 2 * 1000,
                                                                    "max_frame": 8192,
                                                                    "mirror": False,
                                                                    "offset_ms": 0,
                                                                    "rate": 1.0,
                                                                    "random": False})
    save_osu_file(beatmap_meta,
                  beatmap_meta.convertor.objects_to_array(objs, beatmap_meta)[0],
                  map_path.replace(".osu", "_convert.osu"),
                  {"Version": "13 dan - convert"})
