import zipfile
import os
import shutil

import eyed3
import gradio as gr
import numpy as np
import requests
import torch
import yaml
from omegaconf import OmegaConf
from reamber.algorithms.playField import PlayField
from reamber.algorithms.playField.parts import *
from reamber.osu.OsuMap import OsuMap

from mug.data.convertor import save_osu_file, parse_osu_file
from mug.data.utils import gridify, remove_intractable_mania_mini_jacks
from mug.diffusion.ddim import DDIMSampler
from mug.diffusion.diffusion import DDPM
from mug.util import instantiate_from_config, feature_dict_to_embedding_ids, \
    load_audio_without_cache

# rs:Rank Status (default:Ranked)
# sr:star rank
# cj:Chordjack
# sta:stamina
# ss:str
# js:jumpstream
# hs:handstream
# jsp:jackspeed
# tech:technical
# ln, rc, hb, ln_ratio

theme = gr.themes.Soft().set(
    block_title_text_size='*text_xxs',
    checkbox_label_text_size='*text_xxs'
)
theme = gr.themes.Soft()

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

# TODO: make configurable
config = OmegaConf.load("models/ckpt/model.yaml")
model = load_model_from_config(config, "models/ckpt/model.ckpt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
template_path = "data/template.osu"
output_path = "outputs/beatmaps/"
sampler = DDIMSampler(model)

def getHeight(y, ratio):
    left = 1
    right = y
    while(left<=right):
        if left == right:
            return left
        mid = int((left+right)/2)
        res = (86*np.ceil(y/mid)-3)/mid
        if ratio > res:
            right = mid
        elif ratio < res:
            left = mid+1
        else:
            return mid

def generate_feature_dict(audioPath, audioTitle, audioArtist, 
                 rss, rs, srs, sr, etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss, ssss, sssc, jss, js, jsss, jssc,
                    hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc, techs, tech, techss, techsc, mts, lnrs, mapType, lnr,  count, step, scale, rm_jacks, auto_snap):

    print(audioPath)
    feature_dict = {}
    def add_value_if(condition, key, val):
        if condition:
            feature_dict[key] = val

    add_value_if(rss, 'rank_status', 'ranked' if rs == 'ranked/stable' else rs)
    add_value_if(srs, 'sr', sr)
    add_value_if(etts, 'ett', ett)

    patterns = [
        (cjs, 'Chordjack', cj, cjss, cjsc),
        (stas, 'Stamina', sta, stass, stasc),
        (sss, 'Stream', ss, ssss, sssc),
        (jss, 'Jumpstream', js, jsss, jssc),
        (hss, 'Handstream', hs, hsss, hssc),
        (jsps, 'Jackspeed', jsp, jspss, jspsc),
        (techs, 'Technical', tech, techss, techsc)
    ]
    for pattern_switch, pattern_name, pattern_value, pattern_score_switch, pattern_score_value in patterns:
        add_value_if(pattern_switch, pattern_name.lower(), pattern_value.startswith("enhance"))
        # add_value_if(pattern_switch, pattern_name.lower(), pattern_name)
        add_value_if(pattern_score_switch, pattern_name.lower() + "_ett", pattern_score_value)

    if mts:
        feature_dict['ln'] = feature_dict['rc'] = feature_dict['hb'] = 0
        feature_dict[mapType] = 1

    add_value_if(lnrs, 'ln_ratio', lnr)
    return feature_dict

def updatePrompt(*args):
    feature_dict = generate_feature_dict(*args)
    return gr.update(value=str(feature_dict))

def parse_feature(batch_size, feature_dicts, feature_yaml, model: DDPM):
    features = []
    for i in range(batch_size):
        cur_dict = feature_dicts
        print(f"Feature {i}: {cur_dict}")
        features.append(feature_dict_to_embedding_ids(cur_dict, feature_yaml))
    feature = torch.tensor(np.asarray(features), dtype=torch.float32,
                           device=model.device)
    return model.model.cond_stage_model(feature)


def startMapping(audioPath, audioTitle, audioArtist, 
                 rss, rs, srs, sr, etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss, ssss, sssc, jss, js, jsss, jssc,
                 hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc, techs, tech, techss, techsc, mts, lnrs, mapType, lnr,  count, step, scale, rm_jacks, auto_snap,
                 progress=gr.Progress()):

    torch.cuda.empty_cache()
    feature_dict = generate_feature_dict(
        audioPath, audioTitle, audioArtist, rss, rs, srs, sr,
        etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss, ssss, sssc,
        jss, js, jsss, jssc, hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc,
        techs, tech, techss, techsc, mts, lnrs, mapType, lnr, count, step, scale, rm_jacks, auto_snap
    )
    feature_yaml = yaml.safe_load(open("configs/mug/mania_beatmap_features.yaml"))

    print(feature_dict, rm_jacks, auto_snap)

    with torch.no_grad():
        uc = None

        for _ in progress.tqdm([0], desc='Process prompts'):
            if scale != 1.0:
                uc = parse_feature(count, {}, feature_yaml, model)

            c = parse_feature(count, feature_dict, feature_yaml, model)

        max_test_train_length_ratio = 1

        for _ in progress.tqdm([0], desc='Process audio'):
            dataset = config.data.params.common_params

            # dataset.max_audio_frame = 16384
            # model.z_length = 256

            audio_hop_length = dataset.n_fft // 4
            audio_frame_duration = audio_hop_length / dataset.sr

            # support 12 min at most
            audio = load_audio_without_cache(audioPath, dataset.n_mels, dataset.n_fft // 4,
                                         dataset.n_fft, dataset.sr,
                                         audio_frame_duration * dataset.max_audio_frame * max_test_train_length_ratio)
            t = audio.shape[1]
            if t < dataset.max_audio_frame:
                audio = np.concatenate([
                    audio,
                    np.zeros((dataset.n_mels, dataset.max_audio_frame - t), dtype=np.float32)
                ], axis=1)
            elif t > dataset.max_audio_frame:
                # if t > dataset.max_audio_frame * max_test_train_length_ratio
                audio = audio[:, :dataset.max_audio_frame]
            w = torch.tensor(
                np.stack([audio for _ in range(count)]),
                           dtype=torch.float32).to(model.device)
            w = model.model.wave_model(w)

        shape = None
        samples_ddim, _ = sampler.sample(S=step,
                                         c=c, w=w,
                                         batch_size=count,
                                         shape=shape,
                                         verbose=True,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=0.0,
                                         tqdm_class=progress.tqdm)

        x_samples_ddim = model.model.decode(samples_ddim).cpu().numpy()

        save_name = f"{audioArtist} - {audioTitle}"
        save_dir = os.path.join(output_path, save_name)
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
        convertor_params = {
            "frame_ms": audio_frame_duration * dataset.audio_note_window_ratio * 1000,
            "max_frame": dataset.max_audio_frame // dataset.audio_note_window_ratio
        }

        def custom_gridify(hit_objects):
            if rm_jacks:
                hit_objects = remove_intractable_mania_mini_jacks(hit_objects)
            new_hit_objects, bpm, offset = gridify(hit_objects)
            if auto_snap:
                return bpm, offset, new_hit_objects
            else:
                return bpm, offset, hit_objects

        previews = []

        for i, x_sample in enumerate(progress.tqdm(x_samples_ddim, desc='Post process charts')):
            convertor_params = convertor_params.copy()
            convertor_params["from_logits"] = True
            _, beatmap_meta = parse_osu_file(template_path, convertor_params)
            output_name = f"audio{os.path.splitext(audioPath)[-1]}"
            shutil.copyfile(audioPath, os.path.join(save_dir, output_name))
            version = f"AI v{i + 1} ({config.version})"
            creator = "MuG Diffusion"
            file_name = f"{audioArtist} - {audioTitle} ({creator}) [{version}].osu".replace("/", "")
            file_path = os.path.join(save_dir, file_name)

            save_osu_file(beatmap_meta, x_sample,
                          path=file_path,
                          override={
                              "Creator": f"{creator} v{config.version}",
                              "Version": version,
                              "AudioFilename": output_name,
                              "Title": audioTitle,
                              "TitleUnicode": audioTitle,
                              "Artist": audioArtist,
                              "ArtistUnicode": audioArtist,
                          }, gridify=custom_gridify)

            # reamber generate example
            m = OsuMap.read_file(file_path)
            pf = (
                    PlayField(m=m, duration_per_px=5, padding=40) + 
                    PFDrawBpm() + 
                    PFDrawBeatLines() + 
                    PFDrawColumnLines() + 
                    PFDrawNotes() + 
                    PFDrawOffsets()
            )
            originalHeight = pf.export().height
            processedHeight = getHeight(originalHeight, float(3.3))
            pic = pf.export_fold(max_height=processedHeight)
            previews.append(pic)

        # package
        output_osz_path = os.path.join(output_path, save_name + ".osz")
        with zipfile.ZipFile(output_osz_path, 'w') as f:
            for p in os.listdir(save_dir):
                f.write(os.path.join(save_dir, p), arcname=p)


    return [
        gr.update(interactive=True),
        gr.update(value=previews, visible=True),
        gr.update(value=output_osz_path, visible=True)
    ]


if __name__ == "__main__":
    reamberExample = r"Gram - Nibelungen (pieerre) [LN bukbuk].osu"
    # with open("imageView.html", "r") as f:
    #     _html = f.read()
    #     html = _html.replace("\n", " ")
    # print(_html)
    prompt_dir = 'configs/mapping_config/'
    feature_dicts = []
    feature_dicts.append(yaml.safe_load(open(os.path.join(prompt_dir, f"feature_1.yaml"))))
    #print(feature_dicts)
    #startMapping('1', '1', '1', 'ranked', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    #startMapping('D:\CloudMusic\Jianwuhongxiu.mp3','JianwuHongxiu','Luo jiyi','ranked',3,0,1,0,1,1,0,0,0,1,0,0)
    with gr.Blocks() as webui:

        with gr.Row():
            with gr.Column(scale=1):
                # audioPath = DirectAudioPathComponent(label="Audio file", info="drop audio here", type="filepath")
                audioPath = gr.Audio(label="Audio file", info="drop audio here", type="filepath")

            with gr.Column(scale=1):
                audioTitle = gr.Textbox(label="Audio title", lines=1)
                audioArtist = gr.Textbox(label="Audio artist", lines=1)
            def on_change_audio(x):
                try:
                    path = audioPath.base64_to_temp_file_if_needed(x['data'], x['name'])
                    audio_file = eyed3.load(path)
                    audio_artist = audio_file.tag.artist
                    audio_title = audio_file.tag.title
                except:
                    audio_title = x['name'].split('.')[0]
                    audio_artist = ""
                return [
                    gr.update(value=audio_title),
                    gr.update(value=audio_artist),
                ]
            audioPath.change(on_change_audio, inputs=audioPath, outputs=[audioTitle, audioArtist],
                             preprocess=False, postprocess=False)

        with gr.Row():

            with gr.Column(scale=1):

                with gr.Accordion("Diffculty", open=True):

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            sr_switch = gr.Checkbox(label="star rating (osu!)", value=True)
                            ett_switch = gr.Checkbox(label="MSD score (Etterna)")
                        with gr.Column(scale=3, min_width=100):
                            sr = gr.Slider(1, 8, value=4, label="star rating (osu!)")
                            ett = gr.Slider(5, 35, value=20, label="MSD (Etterna)", visible=False)
                    def etts_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def srs_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    ett_switch.select(etts_switch, None, ett)
                    sr_switch.select(srs_switch, None, sr)

                with gr.Accordion("Rice & long note", open=True):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            maptype_switch = gr.Checkbox(label="map type")
                            lnr_switch = gr.Checkbox(label="ln ratio")
                        with gr.Column(scale=3, min_width=100):
                            mapType = gr.Radio(["Rice (LN < 10%)", "Long Note (LN > 40%)", "Hybrid (10% < LN < 70%)"],
                                                show_label=False, value="Rice (LN < 10%)", visible=False)
                            lnr = gr.Slider(0, 1, value=0.0, label="ln ratio", visible=False,
                                        info="ln ratio of the map, 0 for rice only, 1 for FULL LN")
                    def mts_switch(evt:gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def lnrs_switch(evt:gr.SelectData):
                        return gr.update(visible=evt.selected)
                    maptype_switch.select(mts_switch, None, mapType)
                    lnr_switch.select(lnrs_switch, None, lnr)

                with gr.Accordion("Special", open=True):
                    rm_jacks = gr.Checkbox(label="remove intractable mini jacks",
                                           info="recommend when generating stream patterns",
                                           value=True)
                    auto_snap = gr.Checkbox(label="snapping to grids automatically",
                                            info="recommend when there are no bpm changes",
                                            value=True)
                    # aftergen = gr.CheckboxGroup(
                    #     ['remove intractable mini jacks (recommend when generating stream charts)',
                    #      'snapping to grids automatically (recommend when there are no bpm changes)'],
                    #     show_label=False)

                with gr.Accordion("Model configurations", open=True):
                    count = gr.Slider(1, 16, value=4.0, label="Count", info="number of maps",
                                      step=1.0)
                    step = gr.Slider(100, 1000, value=200, label="Step",
                                     info="step of diffusion process", step=1.0)
                    scale = gr.Slider(1, 30, value=5.0, label="CFG scale",
                                      info="prompts matching degree")

            with gr.Column(scale=1):

                with gr.Accordion("Pattern", open=True):

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            rs_switch = gr.Checkbox(label="style", value=True, elem_id="lbox")
                        with gr.Column(scale=3, min_width=100):
                            rs = gr.Radio(['ranked/stable', 'loved', 'graveyard'], \
                                            value='ranked/stable', show_label=False)
                    def rss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    rs_switch.select(rss_switch, None, rs)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            cj_switch = gr.Checkbox(label="chordjack")
                            cj_score_switch = gr.Checkbox(label="chordjack MSD")
                        with gr.Column(scale=3, min_width=100):
                            cj = gr.Radio(['enhance chordjack', 'inhibit chordjack'], value='enhance chordjack',
                                        visible=False, show_label=False)
                            cj_score = gr.Slider(5, 35, value=17, label="chordjack MSD:", visible=False)
                    def cje_switch(evt:gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def cjss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    cj_switch.select(cje_switch, None, cj)
                    cj_score_switch.select(cjss_switch, None, cj_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            sta_switch = gr.Checkbox(label="stamina")
                            sta_score_switch = gr.Checkbox(label="stamina MSD")
                        with gr.Column(scale=3, min_width=100):
                            sta = gr.Radio(['enhance stamina', 'inhibit stamina'], value='enhance stamina',\
                                        visible=False, show_label=False)
                            sta_score = gr.Slider(5, 35, value=17, label="stamina MSD:", visible=False)
                    def stae_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def stass_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    sta_switch.select(stae_switch, None, sta)
                    sta_score_switch.select(stass_switch, None, sta_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            ss_switch = gr.Checkbox(label="stream")
                            ss_score_switch = gr.Checkbox(label="stream MSD")
                        with gr.Column(scale=3, min_width=100):
                            ss = gr.Radio(['enhance stream', 'inhibit stream'], value='enhance stream',
                                        visible=False, show_label=False)
                            ss_score  = gr.Slider(5, 35, value=17, label="stream MSD:", visible=False)
                    def sse_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def ssss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    ss_switch.select(sse_switch, None, ss)
                    ss_score_switch.select(ssss_switch, None, ss_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            js_switch = gr.Checkbox(label="jumpstream")
                            js_score_switch = gr.Checkbox(label="jumpstream MSD")
                        with gr.Column(scale=3, min_width=100):
                            js = gr.Radio(['enhance jumpstream', 'inhibit jumpstream'], value='enhance jumpstream',
                                        visible=False, show_label=False)
                            js_score = gr.Slider(5, 35, value=17, label="jumpstream MSD:", visible=False)
                    def jse_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def jsss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    js_switch.select(jse_switch, None, js)
                    js_score_switch.select(jsss_switch, None, js_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            hs_switch = gr.Checkbox(label="handsteam")
                            hs_score_switch = gr.Checkbox(label="handstream MSD")
                        with gr.Column(scale=3, min_width=100):
                            hs = gr.Radio(['enhance handstream', 'inhibit handsrteam'], value='enhance handstream',
                                        visible=False, show_label=False)
                            hs_score = gr.Slider(5, 35, value=17, label="handsteam MSD:", visible=False)
                    def hse_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def hsss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    hs_switch.select(hse_switch, None, hs)
                    hs_score_switch.select(hsss_switch, None, hs_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            jsp_switch = gr.Checkbox(label="jackspeed")
                            jsp_score_switch = gr.Checkbox(label="jackspeed MSD")
                        with gr.Column(scale=3, min_width=100):
                            jsp = gr.Radio(['enhance jackspeed', 'inhibit jackspeed'], value='enhance jackspeed',
                                        visible=False, show_label=False)
                            jsp_score = gr.Slider(5, 35, value=17, label="jackspeed MSD:", visible=False)
                    def jspe_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def jspss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    jsp_switch.select(jspe_switch, None, jsp)
                    jsp_score_switch.select(jspss_switch, None, jsp_score)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            tech_switch = gr.Checkbox(label="technical")
                            tech_score_switch = gr.Checkbox(label="technical MSD")
                        with gr.Column(scale=3, min_width=100):
                            tech = gr.Radio(['enhance technical', 'inhibit technical'], value='enhance technical',
                                            visible=False, show_label=False)
                            tech_score = gr.Slider(5, 35, value=17, label="technical MSD:", visible=False)
                    def teche_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    def techss_switch(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)
                    tech_switch.select(teche_switch, None, tech)
                    tech_score_switch.select(techss_switch, None, tech_score)

        inp = [audioPath, audioTitle, audioArtist, rs_switch, rs, sr_switch, sr, ett_switch, ett, cj_switch, cj, cj_score_switch, cj_score , sta_switch, sta,\
               sta_score_switch, sta_score, ss_switch, ss, ss_score_switch, ss_score, js_switch, js, js_score_switch, js_score, hs_switch, hs,\
               hs_score_switch, hs_score, jsp_switch, jsp, jsp_score_switch, jsp_score, tech_switch, tech, tech_score_switch, tech_score,\
               maptype_switch, lnr_switch, mapType, lnr, count, step, scale, rm_jacks, auto_snap]
        btn = gr.Button('Start Generation', variant='primary')
        out_preview = gr.Gallery(label="Chart overview", visible=True, elem_id='output').style(
            preview=True
        )
        out_preview.style(object_fit='fill')
        out_file = gr.File(label='Output file', visible=False, interactive=False)
        #out.style(preview=True)

        #def displayWindow(num):
        #    return [gr.update(visible=True) for i in range(num)] + [gr.update(visible=False) for j in range(16-num)]
        #btn.click(displayWindow, count, out)
        btn.click(lambda: gr.update(interactive=False), None, btn)
        btn.click(lambda: gr.update(visible=False), None, out_file)
        btn.click(startMapping, inp, [btn, out_preview, out_file])

        #btn.click(display, t, test)

    #webui.css('lbox { font-size:20px; }')

    webui.queue(10).launch(share=False)
'''
    with gr.Blocks() as demo:
        HTML = gr.HTML(value=_html)
    #demo.launch()
'''
