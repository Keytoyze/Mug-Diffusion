import subprocess
import shutil
import sys
sys.path.append(".")
from mug.data.convertor import *
import numpy as np
import json
import matplotlib.pyplot as plt

# Thanks to https://github.com/bobermilk/etterna2osu/ for addressing the offset issue

def process_audios(audio_list):
    for i, audio in enumerate(audio_list):

        if audio.endswith(("_ogg.mp3", "_mp3.mp3")):
            continue
        raw_name = os.path.basename(audio).replace(".", "_")
        target_name = os.path.join(os.path.dirname(audio), f"{raw_name}.mp3")

        # copy_audio = os.path.join(os.path.dirname(audio), f"temp_audio_sox.{audio.split('.')[-1]}")
        # shutil.move()
        try:
            sample_rate=int(subprocess.run(["wine","scripts/sox/sox.exe", "--i", "-r", audio], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)
            # print(">> sample_rate:", sample_rate)
            average_bitrate=subprocess.run(["wine","scripts/sox/sox.exe", "--i", "-B", audio], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode("utf-8").strip()
            # print(">> average_bitrate:", average_bitrate)
            channel_count=int(subprocess.run(["wine","scripts/sox/sox.exe", "--i", "-c", audio], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)
            # print(">> channel_count:", channel_count)

            # print(">> sox...")
            sox_name = os.path.join(os.path.dirname(audio), f"{raw_name}.raw")

            subprocess.run(["wine","scripts/sox/sox.exe", "-v", "0.99", audio, sox_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # maybe try another codec instead of lame
            # -qscale:a is for VBR higher quality, we use -b:a CBR cuz time sensitive
            
            # print(">> ffmpeg...")
            subprocess.run(["ffmpeg", "-f", "s16le",  "-ar", str(sample_rate) ,"-ac", str(channel_count), "-i", sox_name,"-codec:a" ,"libmp3lame" ,"-b:a" , average_bitrate, target_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"success: {audio} -> {target_name}, {i}/{len(audio_list)}")
            os.remove(sox_name)
            os.remove(audio)
        except:
            print(f"Error: {audio}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":

    process_dir = "data/etterna_offset"

    from multiprocessing import Process

    data = []
    audio_list = []

    for root, dirs, files in os.walk(process_dir):
        for file in files:
            if file.endswith(('.mp3', '.ogg')):
                audio = os.path.join(root, file)
                if audio.endswith(("_ogg.mp3", "_mp3.mp3")):
                    continue
                audio_list.append(audio)
    
    print("\n".join(audio_list))
    print(len(audio_list))
    # raise
    pools = []
                
    n = 1
    for i in range(n + 1):
        start = len(audio_list) // n * i
        end = min(len(audio_list), start + len(audio_list) // n)
        if end <= start:
            continue
        pools.append(Process(target=process_audios, args=(audio_list[start:end],)))
    [p.start() for p in pools]
    [p.join() for p in pools]

    for root, dirs, files in os.walk(process_dir):
        for file in files:
            if file.endswith(('.osu')):
                osu_file = os.path.join(root, file)
                temp_file = os.path.join(root, "temp.osu")
                try:

                    with open(temp_file, "w") as f_edit:
                        with open(osu_file) as f_read:
                            start_hitobject = False
                            for line in f_read.readlines():
                                line = line.strip()
                                if line.startswith("AudioFilename:"):
                                    audio = line.split(":")[-1].strip()
                                    audio = audio.replace(".", "_") + ".mp3"
                                    line = f"AudioFilename: {audio}"
                                elif line.startswith("[HitObjects]"):
                                    start_hitobject = True
                                elif start_hitobject:
                                    if "," in line:
                                        params = line.split(",")
                                        params[2] = str(int(float(params[2]) - 26))
                                        if params[3] == '128':
                                            params[5] = str(int(float(params[5]) - 26))
                                        line = ",".join(params)
                                f_edit.write(line + "\n")
                    
                    shutil.move(temp_file, osu_file)
                except:
                    print(osu_file)
