import pandas as pd
import os


def create_list_if_str(param, length):
    if isinstance(param, str):
        return [param] * length
    return param


def densify(mtx):
    try:
        return mtx.todense()
    except AttributeError:
        return mtx


def add_row_and_save(data, fn):
    if os.path.exists(fn):
        df = pd.read_table(fn)
    else:
        df = pd.DataFrame(columns=data.keys())
    new_cols = set(data.keys() - set(df.columns))
    for c in new_cols:
        df[c] = None
    df = df.append(data, ignore_index=True)
    df = df.sort_index(axis=1)
    df.to_csv(fn, sep='\t', index=False)
