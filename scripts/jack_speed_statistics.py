import sys, os
from collections import defaultdict
sys.path.append(os.getcwd())
from mug.data.convertor import *

dirs = [
    # "logs/2023-02-18T01-42-12_mug_diffusion/",
    # "logs/2023-02-18T14-44-56_mug_diffusion/",
    # "logs/2023-02-18T15-38-29_mug_diffusion/",
    # "logs/2023-02-18T18-45-54_mug_diffusion/",
    # "logs/2023-02-20T16-05-53_mug_diffusion/",
    # "logs/2023-02-21T22-30-55_mug_diffusion/",
    # "logs/2023-02-22T16-02-43_mug_diffusion/",
    # "logs/2023-02-24T17-14-56_mug_diffusion/",
    # "logs/2023-02-26T15-55-17_mug_diffusion/",

    # "logs/2023-03-13T18-06-59_mug_diffusion/",
    # "logs/2023-03-13T23-11-26_mug_diffusion/",
    # "logs/2023-03-14T23-45-51_mug_diffusion",
    # "logs/2023-03-15T15-13-00_mug_diffusion",
    # "logs/2023-03-15T17-37-56_mug_diffusion",

    "logs/2023-03-19T23-19-39_mug_diffusion"
]

def jack_speed_count(osu_path):
    hit_objects, _ = parse_osu_file(osu_path, None)
    key_to_objects = defaultdict(lambda: [])
    column_width = int(512 / 4)

    for line in hit_objects:
        params = line.split(",")
        start = int(float(params[2]))
        column = int(int(float(params[0])) / column_width)
        key_to_objects[column].append(start)
    
    count = 0
    for k in key_to_objects:
        starts = sorted(key_to_objects[k])
        if len(starts) <= 1:
            continue
        for i in range(len(starts) - 1):
            if starts[i + 1] - starts[i] < 100:
                count += 1
    return count



for p in dirs:
    beatmap_path = os.path.join(p, "beatmaps")
    counts = sorted(os.listdir(beatmap_path), key=int)
    real_jack_counts = []
    ai_jack_counts = []
    for i in counts:
        path = os.path.join(beatmap_path, i)
        real_count = 0
        ai_count = 0
        for set_name in os.listdir(path):
            set_path = os.path.join(path, set_name)
            for osu_name in os.listdir(set_path):
                if osu_name.endswith("_step=0.osu"):
                    osu_path = os.path.join(set_path, osu_name)
                    ai_count += jack_speed_count(osu_path)
                    
                if (osu_name.endswith(".osu") and "_step=" not in osu_name):
                    osu_path = os.path.join(set_path, osu_name)
                    real_count += jack_speed_count(osu_path)
                    # print(set_path, jack_speed_count(osu_path))

        real_jack_counts.append(real_count)
        ai_jack_counts.append(ai_count)
    print(p)
    print(ai_jack_counts)
    print(real_jack_counts)