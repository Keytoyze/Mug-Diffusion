import librosa
from mug.data.convertor import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tmp_dir = "data/malody"

    data = []

    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            if file.endswith(('.osu')):
                p = os.path.join(root, file)
                hit_objects, meta = parse_osu_file(p, None)
                hit_objects = sorted(hit_objects, key=lambda x: float(x.split(",")[2]))

                y, sr = librosa.load(meta.audio,
                                     duration=20)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                beat_time = librosa.frames_to_time(beats, sr=sr)

                offsets = []
                for t in beat_time:
                    t = t * 1000
                    cur_diff = 1000000
                    for line in hit_objects:
                        note_t = float(line.split(",")[2])
                        if abs(note_t - t) < abs(cur_diff):
                            cur_diff = note_t - t
                        if note_t - t > 50:
                            break
                    if abs(cur_diff) <= 50:
                        offsets.append(cur_diff)

                if len(offsets) > 0:
                    data.append(np.mean(offsets))

            plt.clf()
            plt.hist(data, bins=20)
            plt.savefig("result.png")