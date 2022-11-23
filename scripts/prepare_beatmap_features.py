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

from mug.data.convertor import *
import matplotlib.pyplot as plt


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


def invoke_osu_tools(beatmap_path, osu_tools):
    cmd = [
        'dotnet',
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


def get_star(path, osu_tools, update_dict):
    sr_old = update_dict.get("sr", None)
    if sr_old is not None:
        return False
    sr = invoke_osu_tools(path, osu_tools)
    update_dict['sr'] = sr
    return True


def get_ln_ratio(path, update_dict):
    ln_ratio = update_dict.get("ln_ratio", None)
    if ln_ratio is not None:
        return False

    ob, _ = parse_osu_file(path, None)

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
    return True


def prepare_features(beatmap_txt, features_yaml, osu_tools):
    features_yaml = yaml.safe_load(open(features_yaml))
    conn = sqlite3.connect(os.path.join(os.path.dirname(beatmap_txt), 'feature.db'))
    type_map = {
        'numeric': 'REAL DEFAULT 0.0',
        'category': 'TEXT DEFAULT NULL',
        'bool': 'INT DEFAULT -1'
    }
    create_table(conn, "Feature", ['name TEXT', 'set_name TEXT'] + list(map(
        lambda x: x['name'].split(",")[-1].strip() + ' ' + type_map[x['type']],
        features_yaml
    )), ['name', 'set_name'])
    stars = []
    ln_ratios = []

    for line in tqdm(list(open(beatmap_txt))):
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

            update = get_star(path, osu_tools, update_dict) or update
            update = get_ln_ratio(path, update_dict) or update

            stars.append(update_dict['sr'])
            ln_ratios.append(update_dict['ln_ratio'])
        except:
            traceback.print_exc()
            print(update_dict)
            continue

        if update:
            insert_or_replace(conn, "Feature", [update_dict])

            conn.commit()

    stars = np.asarray(stars)
    stars = np.clip(stars, 1, 8)
    plt.hist(stars, bins=np.arange(1, 8.1, 0.2))
    plt.show()

    plt.hist(ln_ratios, bins=10)
    plt.show()



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

    opt, _ = parser.parse_known_args()

    prepare_features(opt.beatmap_txt, opt.features_yaml, opt.osu_tools)
