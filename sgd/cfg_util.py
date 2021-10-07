import numpy as np


def vectorize_config(cfg):
    cfg_list = []
    for i in range(len(cfg.keys())):
        cfg_list.append(cfg["p_{}".format(i)])
    return np.array(cfg_list)


def symlog(x, symlog_scale):
    vy_min = symlog_scale['vy_min']
    vy_max = symlog_scale['vy_max']
    vx_min = symlog_scale['vx_min']
    vx_max = symlog_scale['vx_max']
    sx = np.sign(x)
    vx = np.abs(x)
    if vx >= vx_min:
        a = (np.log(vy_max) - np.log(vy_min)) / (vx_max - vx_min)
        b = np.log(vy_min) - a * vx_min
        vy = np.exp(a * vx + b)
    else:
        vy = vy_min / vx_min * vx
    return sx * vy


def symlog_vector(x, symlog_scale):
    return np.array([symlog(xi, symlog_scale) for xi in x])
