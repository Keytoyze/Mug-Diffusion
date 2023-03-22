import os.path
import shutil
import string

from tqdm import tqdm
import sys
sys.path.append(".")

from mug.data.convertor import *

valid_chars = "-_.()[]' %s%s" % (string.ascii_letters, string.digits)


def slugify(text):
    # return "".join(c for c in text if c in valid_chars)
    return text

def safe_copy(src_file, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    new_path = os.path.join(dest_dir, slugify(os.path.basename(src_file)))
    if os.path.isfile(new_path):
        return new_path
    shutil.copyfile(src_file, new_path)
    return new_path


def prepare_local_beatmaps(song_dir, mode_int, out_dir, cs):
    raw_set_names = os.listdir(song_dir)
    print(len(raw_set_names))
    set_names = set()
    for x in raw_set_names:
        try:
            p = os.path.getmtime(os.path.join(song_dir, x))
            if p >= 1679127186:
                set_names.add(x)
        except:
            pass
    print(len(set_names))
    results = []
    for set_name in tqdm(set_names):
        try:
            file_names = os.listdir(os.path.join(song_dir, set_name))
            for file_name in file_names:
                if file_name.endswith(".osu"):
                    path = os.path.join(song_dir, set_name, file_name)
                    hit_mode = False
                    hit_cs = True
                    parsed = 0
                    audio_name = None
                    try:
                        with open(path, encoding='utf-8') as f:
                            for line in f:
                                line = line.lower().strip()
                                if line.startswith("mode"):
                                    hit_mode = int(read_item(line)) == mode_int
                                    parsed += 1
                                elif line.startswith("circlesize"):
                                    hit_cs = cs is None or float(cs) == float(read_item(line))
                                    parsed += 1
                                elif line.startswith("audiofilename"):
                                    audio_name = read_item(line)
                                    audio_name = os.path.join(song_dir, set_name, audio_name)
                                if parsed == 2 and audio_name is not None and os.path.isfile(
                                        audio_name):
                                    break
                    except:
                        pass
                    if hit_mode and hit_cs and os.path.isfile(audio_name):
                        # set_name = set_name.replace("&", "")
                        # file_name = file_name.replace("&", "")
                        new_dir = os.path.join(out_dir, slugify(set_name))
                        new_path = safe_copy(path, new_dir)
                        safe_copy(audio_name, new_dir)
                        results.append(new_path + "\n")
        except:
            continue

    with open(os.path.join(out_dir, "beatmap.txt"), "w", encoding='utf-8') as f:
        f.writelines(results)


if __name__ == '__main__':
    import sys, argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--song_dir',
                        '-s',
                        type=str)
    parser.add_argument('--mode_int',
                        '-m',
                        type=int)
    parser.add_argument('--out_dir',
                        '-o',
                        type=str)
    parser.add_argument('--cs',
                        default=-1,
                        type=float)

    opt, _ = parser.parse_known_args()

    prepare_local_beatmaps(opt.song_dir, opt.mode_int, opt.out_dir, opt.cs)
