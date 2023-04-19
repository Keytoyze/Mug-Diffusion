import time

import numpy as np
from sklearn.linear_model import LinearRegression


def parse_hit_objects(line, column_width):
    if line is None:
        return None, None, None
    params = line.split(",")
    column = int(int(float(params[0])) / column_width)
    start_time = float(params[2])
    return start_time, column, None if int(params[3]) != 128 else float(params[5].split(":")[0])


def test_timing(time_list, test_bpm, test_offset, div, refine):
    cur_offset = test_offset
    cur_bpm = test_bpm

    epsilon = 10
    gap = 60 * 1000 / (test_bpm * div)
    delta_time_list = time_list - test_offset
    meter_list = delta_time_list / gap
    meter_list_round = np.round(meter_list)
    timing_error = np.abs(meter_list - meter_list_round)
    valid = (timing_error < epsilon / gap).astype(np.int32)
    valid_count = np.sum(valid)

    if valid_count >= 2 and refine:
        rgs = LinearRegression(fit_intercept=True)
        rgs.fit(meter_list_round.reshape((-1, 1)), time_list, sample_weight=valid)
        if not np.isinf(rgs.coef_) and not np.isnan(rgs.coef_) and rgs.coef_[0] != 0:
            cur_offset = rgs.intercept_
            cur_bpm = 60000 / rgs.coef_[0] / 4

            while cur_bpm < 150:
                cur_bpm = cur_bpm * 2
            while cur_bpm >= 300:
                cur_bpm = cur_bpm / 2

    # valid_ratio = valid_count
    valid_ratio = valid_count / test_bpm
    return valid_ratio, valid, cur_bpm, cur_offset


def timing(time_list, verbose=True):
    offset = time_list[0]

    best_bpm = None
    best_offset = None
    best_valid_ratio = -1

    # find the best bpm when offset = first time
    st = time.time()
    for test_bpm in np.arange(150, 300, 0.1):\

        valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, test_bpm, offset, div=1,
                                                              refine=False)

        if valid_ratio > best_valid_ratio:
            valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, test_bpm, offset,
                                                                  div=1,
                                                                  refine=True)
            best_valid_ratio = valid_ratio
            best_bpm = cur_bpm
            best_offset = cur_offset
            if verbose:
                print(f"[valid: {valid_ratio} / {len(valid)}] bpm {test_bpm} -> {cur_bpm}, "
                f"offset {offset} -> {cur_offset}")

        # find the best offset when bpm = best bpm
        gap = 60000 / cur_bpm
        for test_offset in np.arange(best_offset, best_offset - gap, -gap / 4):

            valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, cur_bpm,
                                                                  test_offset,
                                                                  div=1,
                                                                  refine=False)
            if valid_ratio > best_valid_ratio:
                valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, cur_bpm,
                                                                      test_offset,
                                                                      div=1,
                                                                      refine=True)
                best_valid_ratio = valid_ratio
                best_bpm = cur_bpm
                best_offset = cur_offset
                if verbose:
                    print(f"[valid: {valid_ratio} / {len(valid)}] bpm {best_bpm} -> {cur_bpm}, "
                    f"offset {offset} -> {cur_offset}")

    _, valid_8, best_bpm, best_offset = test_timing(time_list, best_bpm, best_offset, div=16,
                                                    refine=False)
    _, valid_6, best_bpm, best_offset = test_timing(time_list, best_bpm, best_offset, div=6,
                                                    refine=False)
    valid = np.clip(valid_6 + valid_8, 0, 1)

    if verbose:
        print("Test time:", time.time() - st)
        print(f"Final bpm: {best_bpm}, offset: {best_offset}")
        print(f"Final valid: {np.sum(valid)} / {len(valid)}")
        print(f"Invalid: {time_list[valid == 0]}")

    return best_bpm, best_offset

    # rgs = LinearRegression(fit_intercept=True)
    # rgs.fit(np.asarray(meters).reshape((-1, 1)), times[:i + 1])

epsilon = 10

def gridify(hit_objects, verbose=True):
    key_count = 4  # TODO
    column_width = int(512 / key_count)
    times = []
    for line in hit_objects:
        st, _, _ = parse_hit_objects(line, column_width)
        times.append(st)
    times = np.asarray(times, dtype=np.float32)
    bpm, offset = timing(times, verbose)

    def format_time(t):
        for div in [1, 2, 4, 3, 6, 8, 16, 32]:
            gap = 60 * 1000 / (bpm * div)
            meter = (t - offset) / gap
            meter_round = round(meter)
            timing_error = abs(meter - meter_round)
            if timing_error < epsilon / gap:
                return str(int(meter_round * gap + offset))
        return str(int(t))

    new_hit_objects = []
    for line in hit_objects:
        elements = line.split(",")
        elements[2] = format_time(int(elements[2]))
        if int(elements[3]) == 128:
            e = elements[5].split(":")
            e[0] = format_time(int(e[0]))
            elements[5] = ":".join(e)
        new_hit_objects.append(",".join(elements))
    return new_hit_objects, bpm, offset


def remove_intractable_mania_mini_jacks(hit_objects, verbose=True, jack_interval=90):
    key_count = 4  # TODO
    column_width = int(512 / key_count)
    new_hit_objects = [x for x in hit_objects]

    def has_ln(start_index, column, time):
        i = start_index - 1
        while i >= 0:
            start_time, c, end_time = parse_hit_objects(new_hit_objects[i], column_width)
            i -= 1
            if end_time is None or start_time is None:
                continue
            if c == column and start_time <= time:
                return end_time >= time - 50
        return False


    def get_notes_idx_in_interval(start_index, time, interval, column, search_previous,
                                  search_latter):
        result = []
        i = start_index - 1
        if search_previous:
            while i >= 0:
                st, c, _ = parse_hit_objects(new_hit_objects[i], column_width)
                if st is not None:
                    if abs(st - time) <= interval:
                        if c == column or column < 0:
                            result.append((i, st, c))
                    else:
                        break
                i -= 1
        if search_latter:
            i = start_index + 1
            while i < len(new_hit_objects):
                st, c, _ = parse_hit_objects(new_hit_objects[i], column_width)
                if st is not None:
                    if abs(st - time) <= interval:
                        if c == column or column < 0:
                            result.append((i, st, c))
                    else:
                        break
                i += 1
        return result

    for i in range(len(new_hit_objects)):
        start_time, column, end_time = parse_hit_objects(new_hit_objects[i], column_width)

        previous_jacks = get_notes_idx_in_interval(i, start_time, jack_interval, column,
                                                   search_previous=True, search_latter=False)
        if len(previous_jacks) != 0:
            # Detect jacks!
            # Step 1: judge if it's an end of streams. If so, ignore it.
            notes_after_it = get_notes_idx_in_interval(i, start_time, jack_interval * 2, -1,
                                                       search_previous=False,
                                                       search_latter=True)
            count_notes_after_it = 0
            for n in notes_after_it:
                if abs(n[1] - start_time) >= epsilon:
                    count_notes_after_it += 1
            if count_notes_after_it == 0:
                if verbose:
                    print(f"Ignore: {start_time}, {column}")
                continue

            # Step 2: try to move the notes to other columns.
            # Priority: latter note > previous note, same side > other sides
            success = False
            for (is_ln, try_move_index, try_move_t, try_move_src_column) in [
                (end_time is not None, i, start_time, column),
                (False, ) + previous_jacks[0]
            ]:
                if is_ln:
                    continue # we don't want to move LN since it's intractable
                if try_move_src_column == 0 or try_move_src_column == 1:
                    try_move_dst_columns = (1 - try_move_src_column, 2, 3)
                else:
                    try_move_dst_columns = (5 - try_move_src_column, 1, 0)

                for try_move_dst_column in try_move_dst_columns:
                    if has_ln(try_move_index, try_move_dst_column, try_move_t):
                        continue
                    jacks_after_move = len(get_notes_idx_in_interval(
                        try_move_index, try_move_t, jack_interval, try_move_dst_column,
                        search_previous=True, search_latter=True
                    ))
                    if jacks_after_move == 0:
                        success = True
                        if verbose:
                            print(f"Move: {try_move_t}, {try_move_src_column} -> {try_move_dst_column}")

                        elements = new_hit_objects[try_move_index].split(",")
                        elements[0] = str(int(round((try_move_dst_column + 0.5) * column_width)))
                        new_hit_objects[try_move_index] = ",".join(elements)

                        break
                if success:
                    break
            if success:
                continue

            # Step 3: Remove the note that has the more holds
            holds_latter = len(
                get_notes_idx_in_interval(i, start_time, 10, -1, search_previous=True,
                                          search_latter=True)
            ) + 1
            holds_previous = len(
                get_notes_idx_in_interval(previous_jacks[0][0], previous_jacks[0][1],
                                          10, -1, search_previous=True,
                                          search_latter=True)
            ) + 1
            if holds_latter > 1 and holds_latter >= holds_previous and end_time is None:
                if verbose:
                    print(f"Remove: {start_time} | {column} "
                          f"due to the holds: {holds_latter} >= {holds_previous}")
                new_hit_objects[i] = None
            elif holds_previous > 1 and holds_previous >= holds_latter:
                if verbose:
                    print(f"Remove: {previous_jacks[0][1]} | {column} "
                          f"due to the holds: {holds_latter} >= {holds_previous}")
                new_hit_objects[previous_jacks[0][0]] = None
            elif end_time is not None: # LN, remove previous
                if verbose:
                    print(f"Remove: {previous_jacks[0][1]} | {column} "
                          f"due to LN")
                new_hit_objects[previous_jacks[0][0]] = None
            else:
                if verbose:
                    print(f"Remove: {start_time} | {column} "
                          f"for no reason")
                new_hit_objects[i] = None

    return [x for x in new_hit_objects if x is not None]


if __name__ == "__main__":
    from mug.data.convertor import parse_osu_file, save_osu_file
    import sys

    path = sys.argv[-1]

    new_path = path.replace(".osu", "_refine.osu")

    hit_objects, meta = parse_osu_file(
        path,
        None)

    new_hit_objects = remove_intractable_mania_mini_jacks(hit_objects)

    override = {
        "Version": "rm jack"
    }

    with open(new_path, "w", encoding='utf8') as f:
        for line in meta.file_meta:
            if override is not None:
                for k, v in override.items():
                    if line.startswith(k + ":"):
                        line = f"{k}: {v}"
                        break
            f.write(line + "\n")

        f.write("[HitObjects]\n")

        for hit_object in new_hit_objects:
            f.write(hit_object + "\n")
