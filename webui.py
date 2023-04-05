import gradio as gr
from io import BytesIO
from PIL import Image
import requests
import yaml
import os 

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

def yaml_remove(keyList):
    for key in keyList:
        if key in feature_dicts[0]:
            feature_dicts[0].pop(key)

def startMapping(audioPath, audioTitle, audioArtist, \
                 rss, rs, srs, sr, etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss, ssss, sssc, jss, js, jsss, jssc,\
                    hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc, techs, tech, techss, techsc, mts, lnrs, mapType, lnr, step, count, scale, after):
    removeList = []

    if rss == False:
        removeList.append('rank_status')
    else:
        if rs == 'ranked/stable':
            feature_dicts[0]['rank_status'] = 'ranked'
        else:
            feature_dicts[0]['rank_status'] = rs

    if srs == False:
        removeList.append('sr')
    else:
        feature_dicts[0]['sr'] = sr
    if etts == False:
        removeList.append('ett')
    else:
        feature_dicts[0]['ett'] = ett

    if cjs == False:
        removeList.append('chordjack')
    else:
        feature_dicts[0]['chordjack'] = True if cj == 'enhance Chordjack' else False
    if cjss == False:
        removeList.append('chordjack_ett')
    else:
        feature_dicts[0]['chordjack_ett'] = cjsc

    if stas == False:
        removeList.append('stamina')
    else:
        feature_dicts[0]['stamina'] = True if sta == 'enhance Stamina' else False
    if stass == False:
        removeList.append('stamina_ett')
    else:
        feature_dicts[0]['stamina_ett'] = stasc

    if sss == False:
        removeList.append('stream')
    else:
        feature_dicts[0]['stream'] = True if ss == 'enhance Stream' else False
    if ssss == False:
        removeList.append('stream_ett')
    else:
        feature_dicts[0]['stream_ett'] = sssc

    if jss == False:
        removeList.append('jumpstream')
    else:
        feature_dicts[0]['jumpstream'] = True if js == 'enhance Jumpstream' else False
    if jsss == False:
        removeList.append('jumpstream_ett')
    else:
        feature_dicts[0]['jumpstream_ett'] = jssc

    if hss == False:
        removeList.append('handstream')
    else:
        feature_dicts[0]['handstream'] = True if hs == 'enhance Handstream' else False
    if hsss == False:
        removeList.append('handstream_ett')
    else:
        feature_dicts[0]['handstream_ett'] = hssc

    if jsps == False:
        removeList.append('jackspeed')
    else:
        feature_dicts[0]['jackspeed'] = True if jsp == 'enhance Jackspeed' else False
    if jspss == False:
        removeList.append('jackspeed_ett')
    else:
        feature_dicts[0]['jackspeed_ett'] = jspsc

    if techs == False:
        removeList.append('technical')
    else:
        feature_dicts[0]['technical'] = True if tech == 'enhance Technical' else False
    if techss == False:
        removeList.append('technical_ett')
    else:
        feature_dicts[0]['technical_ett'] = techsc

    if mts == False:
        removeList.append('ln')
        removeList.append('rc')
        removeList.append('hb')
    else:
        feature_dicts[0]['ln'] = feature_dicts[0]['rc'] = feature_dicts[0]['hb'] = 0
        feature_dicts[0][mapType] = 1
    
    if lnrs == False:
        removeList.append('ln_ratio')
    else:
        feature_dicts[0]['ln_ratio'] = lnr

    if 'remove bullets' in after:
        removeBulltets = True
    
    if 'auto attach' in after:
        autoAttach = True

    yaml_remove(removeList)

    with open(os.path.join(prompt_dir, f"feature_1.yaml"), "w", encoding="utf-8") as f:
        
        yaml.dump(feature_dicts[0], f, sort_keys=False)
    #cmd = "python scripts/mapping.py" + ' --audio ' + audioPath + ' --prompt_dir ' + prompt_dir +\
    #      " --audio_title " + '"' + audioTitle + '"' + " --audio_artist " + '"' +\
    #        audioArtist + '"' + " --n_samples 1"
    #os.system('activate MuG_Diffusion' + '&&' + cmd)
    fin = 'finished!'

    url = 'https://osu.sayobot.cn/qiafan/hiosu.png'
    req = requests.get(url)
    pic = Image.open(BytesIO(req.content))
    return pic

if __name__ == "__main__":
    prompt_dir = 'configs/mapping_config/'
    feature_dicts = []
    feature_dicts.append(yaml.safe_load(open(os.path.join(prompt_dir, f"feature_1.yaml"))))
    #print(feature_dicts)
    #startMapping('1', '1', '1', 'ranked', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    #startMapping('D:\CloudMusic\Jianwuhongxiu.mp3','JianwuHongxiu','Luo jiyi','ranked',3,0,1,0,1,1,0,0,0,1,0,0)
    with gr.Blocks(theme=theme) as webui:
        
        audioPath = gr.Audio(label="Audio", info="drop audio here", type="filepath")
        audioTitle = gr.Textbox(label="audioTitle", info="song title", lines=1)
        audioArtist = gr.Textbox(label="audioArtist", info="artist", lines=1)
        
        with gr.Accordion("Diffculty", open=False):
            with gr.Box() as box2:

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        sr_switch = gr.Checkbox(label="star_rating", value=True)
                        ett_switch = gr.Checkbox(label="score(MSD)")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            sr = gr.Slider(1, 8, value=3, label="star rating", info="between 1 and 10")
                            ett = gr.Slider(5, 35, value=17, label="MSD", info="between 5 and 35", visible=False)
                def etts_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def srs_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                ett_switch.select(etts_switch, None, ett)
                sr_switch.select(srs_switch, None, sr)

        with gr.Accordion("Pattern", open=False):
            with gr.Box() as box3:

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
                            cj_score_switch = gr.Checkbox(label="chordjack score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            cj = gr.Radio(['enhance chordjack', 'inhibit chordjack'], value='enhance chordjack',\
                                        visible=False, show_label=False)
                            cj_score = gr.Slider(5, 35, value=17, label="chordjack MSD score:", visible=False)
                def cje_switch(evt:gr.SelectData):
                    return gr.update(visible=evt.selected)
                def cjss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                cj_switch.select(cje_switch, None, cj)
                cj_score_switch.select(cjss_switch, None, cj_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        sta_switch = gr.Checkbox(label="stamina")
                        sta_score_switch = gr.Checkbox(label="stamina score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            sta = gr.Radio(['enhance stamina', 'inhibit stamina'], value='enhance stamina',\
                                        visible=False, show_label=False)
                            sta_score = gr.Slider(5, 35, value=17, label="stamina MSD score:", visible=False)
                def stae_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def stass_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                sta_switch.select(stae_switch, None, sta)
                sta_score_switch.select(stass_switch, None, sta_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        ss_switch = gr.Checkbox(label="stream")
                        ss_score_switch = gr.Checkbox(label="stream score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            ss = gr.Radio(['enhance stream', 'inhibit stream'], value='enhance stream',\
                                        visible=False, show_label=False)
                            ss_score  = gr.Slider(5, 35, value=17, label="stream MSD score:", visible=False)
                def sse_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def ssss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                ss_switch.select(sse_switch, None, ss)
                ss_score_switch.select(ssss_switch, None, ss_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        js_switch = gr.Checkbox(label="jumpstream")
                        js_score_switch = gr.Checkbox(label="jumpstream score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            js = gr.Radio(['enhance jumpstream', 'inhibit jumpstream'], value='enhance jumpstream',\
                                        visible=False, show_label=False)
                            js_score = gr.Slider(5, 35, value=17, label="jumpstream MSD score:", visible=False)
                def jse_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def jsss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                js_switch.select(jse_switch, None, js)
                js_score_switch.select(jsss_switch, None, js_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        hs_switch = gr.Checkbox(label="handsteam")
                        hs_score_switch = gr.Checkbox(label="handstream score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            hs = gr.Radio(['enhance handstream', 'inhibit handsrteam'], value='enhance handstream',\
                                        visible=False, show_label=False)
                            hs_score = gr.Slider(5, 35, value=17, label="handsteam MSD score:", visible=False)
                def hse_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def hsss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                hs_switch.select(hse_switch, None, hs)
                hs_score_switch.select(hsss_switch, None, hs_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        jsp_switch = gr.Checkbox(label="jackspeed")
                        jsp_score_switch = gr.Checkbox(label="jackspeed score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            jsp = gr.Radio(['enhance jackspeed', 'inhibit jackspeed'], value='enhance jackspeed',\
                                        visible=False, show_label=False)
                            jsp_score = gr.Slider(5, 35, value=17, label="jackspeed MSD score:", visible=False)
                def jspe_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def jspss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                jsp_switch.select(jspe_switch, None, jsp)
                jsp_score_switch.select(jspss_switch, None, jsp_score)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        tech_switch = gr.Checkbox(label="technical")
                        tech_score_switch = gr.Checkbox(label="technical score")
                    with gr.Column(scale=3, min_width=100):
                        with gr.Row():
                            tech = gr.Radio(['enhance technical', 'inhibit technical'], value='enhance technical',\
                                            visible=False, show_label=False)
                            tech_score = gr.Slider(5, 35, value=17, label="technical MSD score:", visible=False)
                def teche_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                def techss_switch(evt: gr.SelectData):
                    return gr.update(visible=evt.selected)
                tech_switch.select(teche_switch, None, tech)
                tech_score_switch.select(techss_switch, None, tech_score)
        
        with gr.Accordion("Rice & long note", open=False):
            with gr.Box() as box4:
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        maptype_switch = gr.Checkbox(label="map type")
                        lnr_switch = gr.Checkbox(label="ln ratio")
                    with gr.Column(scale=3, min_width=100):   
                        mapType = gr.Radio(["Rice (LN ratio < 10%)", "Long Note (LN ratio > 40%)", "Hybrid (10% < LN ratio < 70%)"],\
                                            show_label=False, value="rc (LN ratio < 10%)", visible=False)
                        lnr = gr.Slider(0, 1, value=0.0, label="ln ratio", visible=False,\
                                    info="ln ratio of the map, 0 for rice only, 1 for FULL LN")
                def mts_switch(evt:gr.SelectData):
                    return gr.update(visible=evt.selected)
                def lnrs_switch(evt:gr.SelectData):
                    return gr.update(visible=evt.selected)
                maptype_switch.select(mts_switch, None, mapType)
                lnr_switch.select(lnrs_switch, None, lnr)

        with gr.Accordion("Model arguments", open=False):
            with gr.Box() as box5:
                count = gr.Slider(1, 16, value=4.0, label="Count", info="number of maps", step=1.0)
                step =  gr.Slider(100, 1000, value=200, label="Step", info="step of diffusion model", step=1.0)
                scale = gr.Slider(1, 30, value=5.0, label="Cfg scale", info="Prompts matching degree")
                aftergen = gr.CheckboxGroup(['remove bullets', 'auto attach'], show_label=False)

        inp = [audioPath, audioTitle, audioArtist, rs_switch, rs, sr_switch, sr, ett_switch, ett, cj_switch, cj, cj_score_switch, cj_score , sta_switch, sta,\
               sta_score_switch, sta_score, ss_switch, ss, ss_score_switch, ss_score, js_switch, js, js_score_switch, js_score, hs_switch, hs,\
               hs_score_switch, hs_score, jsp_switch, jsp, jsp_score_switch, jsp_score, tech_switch, tech, tech_score_switch, tech_score,\
               maptype_switch, lnr_switch, mapType, lnr, count, step, scale, aftergen]
        btn = gr.Button('Start Mapping')
        out = gr.Image(label="Map Overview")
        btn.click(startMapping, inp, out)


    #webui.css('lbox { font-size:20px; }')

    webui.launch(share=True)
'''
    with gr.Blocks() as demo:
        with gr.Accordion("See Details"):
            gr.Markdown("lorem ipsum")
    demo.launch()
'''    
