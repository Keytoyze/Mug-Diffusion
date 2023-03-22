import shutil
import string
import json
import subprocess
import sqlite3
import time
import traceback

import numpy as np
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
import minacalc
import sys
import os
sys.path.append(os.getcwd())
from mug.data.convertor import *


def execute_sql(conn: sqlite3.Connection, sql: str, parameters=None):
    # print(sql, parameters, end='   ')
    start = time.time()
    try:
        if parameters is not None:
            result = conn.execute(sql, parameters)
        else:
            result = conn.execute(sql)
    except sqlite3.OperationalError as e:
        print(sql, parameters)
        raise e
    # print(time.time() - start, 's')
    return result


def execute_sql_return_first(conn: sqlite3.Connection, sql: str, parameters):
    cursor = execute_sql(conn, sql, parameters)
    for x in cursor:
        return x
    return None


def execute_many(conn: sqlite3.Connection, sql: str, seq_of_parameters: list = None):
    # print("sql: %s [%d]" % (sql, len(seq_of_parameters)))
    conn.executemany(sql, seq_of_parameters)


def invoke_osu_tools(beatmap_path, osu_tools, dotnet_path='dotnet'):
    cmd = [
        dotnet_path,
        osu_tools
    ]
    cmd.extend(["difficulty", beatmap_path, "-j"])
    result = json.loads(subprocess.check_output(cmd))
    return result['results'][0]['attributes']['star_rating']


def create_table(conn: sqlite3.Connection, table_name: str, columns,
                 primary_keys: list = None):
    if primary_keys is not None:
        columns.append("PRIMARY KEY (%s)" % ", ".join(primary_keys))
    sql = "CREATE TABLE IF NOT EXISTS `%s` (%s)" % (
        table_name,
        ", ".join(columns)
    )
    execute_sql(conn, sql)


def insert_or_replace(conn: sqlite3.Connection, table_name: str, contents: list, or_ignore=False):
    if len(contents) == 0:
        return
    columns = contents[0].keys()
    sql = "INSERT OR " + ("IGNORE" if or_ignore else "REPLACE") + " INTO `%s` (%s) VALUES (%s)" % (
        table_name, ", ".join(columns),
        ", ".join(["?"] * (len(columns)))
    )

    seq_of_params = [
        [model[column] for column in columns]
        for model in contents
    ]
    execute_many(conn, sql, seq_of_params)


def get_star(path, osu_tools, update_dict, dotnet_path):
    sr_old = update_dict.get("sr", None)
    if sr_old is not None:
        return False
    sr = invoke_osu_tools(path, osu_tools, dotnet_path)
    update_dict['sr'] = sr
    return True

def get_ob_and_meta(path, update_dict):
    if "_ob" not in update_dict or "_meta" not in update_dict:
        update_dict["_ob"], update_dict["_meta"] = parse_osu_file(path, None)
    return update_dict["_ob"], update_dict["_meta"]

def get_rank_status(path, update_dict, rank_maps):
    rank_status = update_dict.get("rank_status", None)
    if rank_status is not None and rank_status != "NULL":
        return False
    _, meta = get_ob_and_meta(path, update_dict)
    update_dict["rank_status"] = rank_maps.get(meta.set_id, "graveyard")
    return True

def get_ett_scores(path, update_dict):
    ett = update_dict.get("ett", 0)
    if ett != 0:
        return False

    ob, _ = get_ob_and_meta(path, update_dict)
    notes = []

    for line in ob:
        if line.strip() == "":
            continue
        try:
            params = line.split(",")
            start = int(float(params[2]))
            column = int(int(float(params[0])) / int(512 / 4))
            assert column <= 3
            notes.append((start, column))
        except:
            pass

    notes = sorted(notes, key=lambda x: x[0])
    result = minacalc.calc_skill_set(1.0, notes)
    keys = [
        "overall",
        "stream",
        "jumpstream",
        "handstream",
        "stamina",
        "jackspeed",
        "chordjack",
        "technical",
    ]
    result = dict(zip(keys, result))
    result_patterns = result.copy()
    del result_patterns['overall']
    del result_patterns['stamina']
    max_score = max(result_patterns.values())

    update_dict.update({
        "ett": result['overall'],
        "stream_ett": result['stream'],
        "jumpstream_ett": result['jumpstream'],
        "handstream_ett": result['handstream'],
        "jackspeed_ett": result['jackspeed'],
        "chordjack_ett": result['chordjack'],
        "technical_ett": result['technical'],
        "stamina_ett": result['stamina'],
        "stream": int(max_score - result['stream'] <= 1),
        "jumpstream": int(max_score - result['jumpstream'] <= 1),
        "handstream": int(max_score - result['handstream'] <= 1),
        "jackspeed": int(max_score - result['jackspeed'] <= 1),
        "chordjack": int(max_score - result['chordjack'] <= 1),
        "technical": int(max_score - result['technical'] <= 1),
        "stamina": int(max_score - result['technical'] <= 1),
    })

def get_ln_ratio(path, update_dict):
    ln_ratio = update_dict.get("ln_ratio", None)
    if ln_ratio is not None:
        return False

    ob, _ = get_ob_and_meta(path, update_dict)

    ln = 0
    rc = 0
    for l in ob:
        params = l.split(",")
        if int(params[3]) == 128:
            ln += 1
        else:
            rc += 1
    if ln == 0 and rc == 0:
        return False
    ln_ratio = ln / (ln + rc)

    is_rc = ln_ratio < 0.1
    is_ln = ln_ratio >= 0.4
    is_hb = 0.1 <= ln_ratio <= 0.7

    update_dict.update({
        'ln_ratio': ln_ratio,
        'rc': int(is_rc),
        'ln': int(is_ln),
        'hb': int(is_hb)
    })
    assert "ln_ratio" in update_dict
    return True

def get_columns_by_cursor(cursor):
    descriptions = list(cursor.description)
    return [description[0] for description in descriptions]

def ensure_column(conn: sqlite3.Connection, table_name: str,
                  name_type_default):
    columns = get_columns_by_cursor(conn.execute("SELECT * FROM Feature"))
    for name, db_type, default in name_type_default:
        if name not in columns:
            if default is not None:
                statement = ("ALTER TABLE %s ADD COLUMN %s %s DEFAULT `%s`" % (
                    table_name, name, db_type, default
                ))
            else:
                statement = ("ALTER TABLE %s ADD COLUMN %s %s" % (
                    table_name, name, db_type
                ))
            execute_sql(conn, statement)

def prepare_features(beatmap_txt, features_yaml, osu_tools, ranked_map_path, dotnet_path):
    features_yaml = yaml.safe_load(open(features_yaml))
    ranked_maps = {}
    if ranked_map_path is not None:
        with open(ranked_map_path) as f:
            for line in f:
                set_id, status = line.strip().split(" ")
                ranked_maps[int(set_id)] = status

    conn = sqlite3.connect(os.path.join(os.path.dirname(beatmap_txt), 'feature.db'))
    type_map = {
        'numeric': 'REAL',
        'category': 'TEXT',
        'bool': 'INT'
    }
    default_map = {
        'numeric': '0.0',
        'category': 'NULL',
        'bool': '-1'
    }
    create_table(conn, "Feature", ['name TEXT', 'set_name TEXT'], ['name', 'set_name'])

    for x in features_yaml:
        ensure_column(conn, "Feature", [(
            x['name'].split(",")[-1].strip(),
            type_map[x['type']],
            default_map[x['type']]
        )])


    stars = []
    ln_ratios = []

    for line in tqdm(list(open(beatmap_txt, encoding='utf8'))):
        path = line.strip()
        if path == "":
            continue
        name = os.path.basename(path)
        set_name = os.path.basename(os.path.dirname(path))

        update_dict = {
            'name': name,
            'set_name': set_name
        }
        update = False

        try:
            cursor = conn.execute("SELECT * FROM Feature WHERE name = ? AND set_name = ?",
                                  [name, set_name])
            descriptions = list(cursor.description)
            columns = [description[0] for description in descriptions]
            row = cursor.fetchone()
            if row is not None:
                update_dict.update(dict(zip(columns, row)))

            update = get_star(path, osu_tools, update_dict, dotnet_path) or update
            update = get_ln_ratio(path, update_dict) or update
            update = get_rank_status(path, update_dict, ranked_maps) or update
            update = get_ett_scores(path, update_dict) or update

            stars.append(update_dict['sr'])
            ln_ratios.append(update_dict['ln_ratio'])
        except:
            traceback.print_exc()

            if '_ob' in update_dict:
                del update_dict["_ob"]
                del update_dict["_meta"]
            print(update_dict)
            continue

        if update:
            if '_ob' in update_dict:
                del update_dict["_ob"]
                del update_dict["_meta"]
            insert_or_replace(conn, "Feature", [update_dict])

            conn.commit()

    # stars = np.asarray(stars)
    # stars = np.clip(stars, 1, 8)
    # plt.hist(stars, bins=np.arange(1, 8.1, 0.2))
    # plt.show()
    #
    # plt.hist(ln_ratios, bins=10)
    # plt.show()



if __name__ == '__main__':
    import sys, argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--beatmap_txt',
                        '-b',
                        type=str)
    parser.add_argument('--features_yaml',
                        '-f',
                        type=str)
    parser.add_argument('--osu_tools',
                        type=str)
    parser.add_argument('--ranked_map_path',
                        type=str,
                        default=None)
    parser.add_argument('--dotnet_path',
                        type=str, 
                        default='dotnet')

    opt, _ = parser.parse_known_args()

    prepare_features(opt.beatmap_txt, opt.features_yaml, opt.osu_tools, opt.ranked_map_path, 
                     opt.dotnet_path)
