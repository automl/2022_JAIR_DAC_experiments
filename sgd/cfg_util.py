import numpy as np

class LinearScale:

    def __init__(self, vy_min, vy_max):
        self.vy_min = vy_min
        self.vy_max = vy_max

    def domain(self):
        return [0, 1]

    def apply(self, value):
        return (self.vy_max - self.vy_min) * value + self.vy_min


class LogScale:

    def __init__(self, vy_min, vy_max):
        self.vy_min = vy_min
        self.vy_max = vy_max

    def domain(self):
        return [0, 1]

    def apply(self, value):
        log_vy = (np.log(self.vy_max) - np.log(self.vy_min)) * value + np.log(self.vy_min)
        return np.exp(log_vy)


class SymLogScale:

    def __init__(self, vx_min, vx_max, vy_min, vy_max):
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max

    def domain(self):
        return [-self.vx_max, self.vx_max]

    def apply(self, value):
        vx_min = self.vx_min
        vx_max = self.vx_max
        vy_min = self.vy_min
        vy_max = self.vy_max
        sx = np.sign(value)
        vx = np.abs(value)
        if vx >= vx_min:
            a = (np.log(vy_max) - np.log(vy_min)) / (vx_max - vx_min)
            b = np.log(vy_min) - a * vx_min
            vy = np.exp(a * vx + b)
        else:
            vy = vy_min / vx_min * vx
        return sx * vy


def vectorize_config(cfg):
    cfg_list = []
    for i in range(len(cfg.keys())):
        cfg_list.append(cfg["p_{}".format(i)])
    return np.array(cfg_list)


def scale_vector(x, scale):
    return np.array([scale.apply(xi) for xi in x])
