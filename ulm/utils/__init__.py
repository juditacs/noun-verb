def create_list_if_str(param, length):
    if isinstance(param, str):
        return [param] * length
    return param


def densify(mtx):
    try:
        return mtx.todense()
    except AttributeError:
        return mtx
