import os
import sys
import argparse
sys.path.append(os.getcwd())

from mug.data.convertor import *
import hashlib


parser = argparse.ArgumentParser()
parser.add_argument('path',
                    nargs="+",
                    type=str)

opt, _ = parser.parse_known_args()

md5_to_path = {}
paths = []

for path in opt.path:
    paths.extend(open(path).readlines())
for path in paths:
    try:
        path = path.strip()
        if path == "":
            continue
        hit_objects, meta = parse_osu_file(path, None)
        column_width = int(512 / 4)

        notes = []
        for line in hit_objects:
            params = line.split(",")
            start = int(float(params[2]))
            end = None
            column = int(int(float(params[0])) / column_width)
            if int(params[3]) == 128:
                end = int(float(params[5].split(":")[0]))
            notes.append((start, end, column))
        if len(notes) == 0:
            continue
        notes = sorted(notes, key=lambda x: x[0] * 100 + x[-1])
        offset = notes[0][0]
        notes = tuple((x[0] - offset, None if x[1] is None else x[1] - offset, x[2]) for x in notes)
        md5 = int(hashlib.md5(str(notes).encode('utf-8')).hexdigest(), 16)
        if md5 in md5_to_path:
            print(md5_to_path[md5], path)
        else:
            valid = True
            for number in ['1.1', '1.2', '1.3', '1.4', '1.05', '1.15', '1.25', '1.35', '1.45', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95']:
                if f'{number}x' in path or f'x{number}' in path or f'{number}]' in path:
                    valid = False
                    break
                number = number.replace(".", ",")
                if f'{number}x' in path or f'x{number}' in path or f'{number}]' in path:
                    valid = False
                    break
            if valid:
                with open("clean.txt", "a+") as f:
                    f.write(path + "\n")
            else:
                print(path)
        md5_to_path[md5] = path
    except:
        import traceback
        traceback.print_exc()
    

