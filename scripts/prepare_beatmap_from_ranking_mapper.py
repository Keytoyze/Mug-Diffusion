import os
from collections import defaultdict
from mug.data.convertor import *
import sqlite3
from tqdm import tqdm
import time
import requests
import json


session = None
recent_request = None
recent_request_time = 0

DEBUG = False
REQUEST_MIN_INTERVAL = 1


def read_creator(path):
    try:
        for line in open(path, encoding='utf8'):
            if line.startswith("Creator:"):
                return line.split(":")[1].strip()
    except:
        pass
    return None

def request_api(api, method, end_point='https://osu.ppy.sh/api/v2/', params=None, header=None, retry_count=0):
    if params is None:
        params = {}
    if header is None:
        header = {}
    url = end_point + api
    global session, recent_request, recent_request_time, REQUEST_MIN_INTERVAL
    if session is None:
        session = requests.Session()
    recent_request = "{method} {url} params: {params}, headers: {headers}".format(
        method=method, url=url,
        params=params,
        headers=header)
    if DEBUG:
        print(recent_request)

    current_interval = time.time() - recent_request_time
    if current_interval < REQUEST_MIN_INTERVAL:
        time.sleep(REQUEST_MIN_INTERVAL - current_interval)
    recent_request_time = time.time()

    try:
        if method.lower() == 'get':
            response = session.get(url, params=params, timeout=60, headers=header).json()
        else:
            response = session.post(end_point + api, data=params, timeout=60, headers=header).json()
    except Exception as e:
        if retry_count >= 5:
            raise e
        session = None
        print("retry...")
        time.sleep(10 + retry_count * 30)
        return request_api(api, method, end_point, params, header, retry_count + 1)
    recent_request += " -> " + str(response)
    return response


def auth(params, save_name):
    secret = {
        "client_id": 11678,
        "client_secret": "vwdFZCHHhViJD5k8alz0PsPa30DdOgzjlhv4V3We",
        "redirect_uri": "http://keytoix.vip/mania/api/osu-oauth",
        "scope": "public"
    }
    params.update(secret)
    auth_data = request_api('token', 'post', end_point='https://osu.ppy.sh/oauth/',
                            params=params)
    auth_data['expire_time'] = time.time() + auth_data['expires_in'] - 3600
    print("auth success!")
    with open(save_name, 'w') as f:
        json.dump(auth_data, f)
    return auth_data


def get_access_token():
    # get cache token
    auth_data = {}
    auth_cache_name = "scripts/auth.json"
    if os.path.exists(auth_cache_name):
        auth_data = json.load(open(auth_cache_name))
    expire_time = auth_data.get('expire_time', 0)
    if time.time() >= expire_time:
        refresh_token = auth_data.get('refresh_token', None)
        if refresh_token is None:
            # auth first
            # webbrowser.open("http://keytoix.vip/mania/api/osu-oauth")
            print("http://keytoix.vip/mania/api/osu-oauth")
            code = input("Please open the above url, and paste the code: ")
            auth_data = auth({'grant_type': 'authorization_code', 'code': code}, auth_cache_name)
        else:
            # refresh token
            auth_data = auth({'grant_type': 'refresh_token', 'refresh_token': refresh_token},
                             auth_cache_name)
    return auth_data['token_type'] + ' ' + auth_data['access_token']


def request_auth_api(api, method, params):
    access_token = get_access_token()
    header = {'Authorization': access_token}
    return request_api(api, method, params=params, header=header)

if __name__ == "__main__":

    content = json.load(open("ranking_map.json"))
    import matplotlib.pyplot as plt
    import math
    # plt.hist(list(filter(lambda x: x >= 50, map(lambda x: x[1], content.items()))), bins=100)
    # plt.show()
    # raise
    setid = []
    for x in content:
        if content[x] < 50:
            continue
        setid.append(str(x))
    with open("download.txt", "w") as f:
        f.write("\n".join(setid))

    feature_db_path = "data/beatmap_4k/feature.db"
    conn = sqlite3.connect(feature_db_path)

    mapper_to_maps = defaultdict(lambda: [])
    ranking_mappers = defaultdict(lambda: 0)
    for x in tqdm(
            conn.execute("SELECT name, set_name FROM Feature WHERE rank_status == 'ranked'")
                    .fetchall()
    ):
        name, set_name = x
        path = os.path.join('data/beatmap_4k/', set_name, name)
        creator = read_creator(path)
        if creator is not None:
            ranking_mappers[creator] += 1

    items = sorted(ranking_mappers.items(), key=lambda x: -x[1])
    data = request_auth_api("users/7304075/beatmapsets/graveyard", "GET", {"limit": 100})

    sets_to_pc = {}
    for username, count in tqdm(items):
        try:
            uid = request_auth_api(f"users/{username}/", "GET", {})['id']
            data = request_auth_api(f"users/{uid}/beatmapsets/graveyard", "GET", {"limit": 1000})
            for x in data:
                has_mania_4k = False
                for b in x['beatmaps']:
                    if "4K" in b['version']:
                        has_mania_4k = True
                        break
                if has_mania_4k:
                    sets_to_pc[x['id']] = x['play_count']
            with open("ranking_map.json", "w") as f:
                json.dump(sets_to_pc, f, indent=2)
        except:
            pass
