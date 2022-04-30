import json
import os

from dac.abstract import DACEnv
import numpy as np


class DACBenchWrapper(DACEnv):

    def __init__(self, benchmark, policy_space):
        self.env = benchmark.get_environment()
        instances_dict = self.env.get_instance_set()
        self.instance_set = [instances_dict[k] for k in sorted(instances_dict)]
        self._policy_space = policy_space
        # store some info in dict format
        """
        temp_file = '{}.temp'.format(id(benchmark))
        benchmark.save_config(temp_file)
        with open(temp_file, mode='r') as fh:
            conf = json.load(fh)
        os.remove(temp_file)
        self.benchmark_info = {'name': type(benchmark).__name__, 'config': conf}
        """

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human', close=False):
        self.env.render(mode)

    @property
    def conditions(self):
        return self.instance_set

    def conditioned_reset(self, selected_condition):
        self.env.set_instance_set({0: selected_condition})
        return self.env.reset()

    def policy_space(self):
        return self._policy_space
