import json
import os
import shutil
import sys
import zipfile

sys.path.append(os.getcwd())
from mug.data.convertor import *
from dataclasses import dataclass

valid_chars = "-_.()[]' %s%s" % (string.ascii_letters, string.digits)


def slugify(text):
    return "".join(c for c in text if c in valid_chars)


@dataclass
class MalodyBPMStamp:
    time: float
    bpm: float
    beat_value: float


def beat2time(beat_value: float, last_stamp: MalodyBPMStamp) -> float:
    return (beat_value - last_stamp.beat_value) * 60000 / last_stamp.bpm + last_stamp.time


def beat2time_with_bpm(beat_value: float, bpm_list: List[MalodyBPMStamp]) -> float:
    position = len(bpm_list)
    for i, x in enumerate(bpm_list):
        if x.beat_value > beat_value:
            position = i
            break
    position -= 1
    return beat2time(beat_value, bpm_list[position])


def get_beat_value(beat_array):
    return beat_array[0] + float(beat_array[1]) / beat_array[2] + 1


def mc_file_2_osu(template_path, mc_path, out_dir, raw_set_name):
    data = json.load(open(mc_path, encoding='utf8'))
    meta = data['meta']
    if meta['mode'] != 0:
        return
    key = meta['mode_ext']['column']
    if key != 4:
        return
    title = meta['song']['title']
    artist = meta['song']['artist']
    version = meta['version']
    creator = meta['creator']

    # time
    bpm_list = []
    time_list = sorted(data['time'], key=lambda x: get_beat_value(x['beat']))
    for i in range(len(time_list)):
        time_obj = time_list[i]
        cur_beat_value = get_beat_value(time_obj['beat'])
        cur_bpm = time_obj['bpm']
        if i == 0:
            bpm_list.append(MalodyBPMStamp(time=0.0, bpm=cur_bpm, beat_value=cur_beat_value))
        else:
            cur_time = beat2time(cur_beat_value, bpm_list[-1])
            bpm_list.append(MalodyBPMStamp(time=cur_time, bpm=cur_bpm, beat_value=cur_beat_value))

    # note
    note_list = []
    column_width = int(512 / key)
    offset = None
    sound = None

    for x in sorted(data['note'], key=lambda x: get_beat_value(x['beat'])):
        column = x.get('column', None)
        if column is None and 'sound' in x:
            sound = x['sound']
            offset = x.get("offset", 0)
            continue
        start_time = beat2time_with_bpm(get_beat_value(x['beat']), bpm_list)
        column_num = int(round((column + 0.5) * column_width))
        if 'endbeat' in x:
            end_time = beat2time_with_bpm(get_beat_value(x['endbeat']), bpm_list)
            note_list.append((column_num, start_time, end_time))
        else:
            note_list.append((column_num, start_time, None))

    assert offset is not None
    assert sound is not None

    set_name = raw_set_name
    name = f"{os.path.basename(mc_path).replace('.mc', '')}.osu"
    set_dir = os.path.join(out_dir, set_name)
    osu_path = os.path.join(set_dir, name)
    try:
        os.makedirs(set_dir, exist_ok=True)
    except:
        pass
    out_song_path = os.path.join(set_dir, sound)
    in_song_path = os.path.join(os.path.dirname(mc_path), sound)
    if not os.path.exists(out_song_path):
        shutil.copyfile(in_song_path, out_song_path)

    hit_objects = []

    for column_num, start_time, end_time in note_list:
        start_time -= offset
        start_time = int(round(start_time))
        if end_time is not None:
            end_time -= offset
            end_time = int(round(end_time))
            hit_objects.append(f"{column_num},192,{start_time},128,0,{end_time}:0:0:0:0:")
        else:
            hit_objects.append(f"{column_num},192,{start_time},1,0,0:0:0:0:")

    templates = open(template_path).read().split("\n")
    override = {
        "Creator": creator,
        "Version": version,
        "AudioFilename": sound,
        "Title": title,
        "TitleUnicode": title,
        "Artist": artist,
        "ArtistUnicode": artist,
    }

    with open(osu_path, "w", encoding='utf8') as f:
        for line in templates:
            if line.startswith("[HitObjects]"):
                continue
            for k, v in override.items():
                if line.startswith(k + ":"):
                    line = f"{k}: {v}"
                    break
            f.write(line + "\n")

        f.write(f"[TimingPoints]\n{-offset},{60000 / bpm_list[0].bpm},4,2,1,20,1,0\n\n")
        f.write("[HitObjects]\n")

        for hit_object in hit_objects:
            f.write(hit_object + "\n")

if __name__ == '__main__':
    import sys, argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--malody_dir',
                        '-b',
                        type=str)
    parser.add_argument('--output_dir',
                        '-f',
                        type=str)

    opt, _ = parser.parse_known_args()


    tmp_dir = os.path.join(opt.output_dir, "temp")
    for f in os.listdir(opt.malody_dir):
        path = os.path.join(opt.malody_dir, f)
        if path.endswith(".mcz"):
            zipfile.ZipFile(path).extractall(tmp_dir)
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file.endswith(('.mc')):
                        mc_path = os.path.join(root, file)
                        try:
                            mc_file_2_osu("data/template.osu", mc_path, opt.output_dir,
                                          f.replace(".mcz", ""))
                        except:
                            import traceback
                            # raise
                            # traceback.print_exc()
                            print("Error:", path, mc_path)

            shutil.rmtree(tmp_dir)
