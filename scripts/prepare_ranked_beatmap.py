import os
import json
import requests
import time

from tqdm import tqdm

session = None
recent_request = None
recent_request_time = 0

DEBUG = False
REQUEST_MIN_INTERVAL = 1

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
    auth_cache_name = "auth.json"
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

    import sys, argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        '-o',
                        type=str)
    parser.add_argument('--mode_num',
                        '-m',
                        type=int)

    opt, _ = parser.parse_known_args()


    songs = set()
    state = {
        'm': int(opt.mode_num)
    }
    pbar = None
    while True:
        data = request_auth_api('beatmapsets/search/', method='GET', params=state)
        if pbar is None:
            pbar = tqdm(total=data['total'])
        pbar.update(len(data["beatmapsets"]))
        for beatset in data["beatmapsets"]:
            set_id = int(beatset['id'])
            if set_id not in songs:
                with open(opt.output_path, "a+") as f:
                    f.write(f"{set_id} {beatset['status']}\n")
        if 'cursor_string' in data and data['cursor_string'] is not None:
            state["cursor_string"] = data["cursor_string"]
        else:
            break



