import json
import os

from sgd.dac.abstract import DACEnv
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


class StochasticDACBenchWrapper(DACBenchWrapper):

    def __init__(self, benchmark, policy_space, seeds_per_cond, seed=42, same_seeds=False):
        super().__init__(benchmark, policy_space)
        rng = np.random.RandomState(seed)
        self.conditions_w_seed = []
        if same_seeds:
            seeds = [rng.random_integers(1, 2147483647) for _ in range(seeds_per_cond)]
            for i in range(seeds_per_cond):
                for c in super().conditions:
                    self.conditions_w_seed.append((c, seeds[i]))
        else:
            for _ in range(seeds_per_cond):
                for c in super().conditions:
                    self.conditions_w_seed.append((c, rng.random_integers(1, 2147483647)))
        self.rng_instance = None

    def step(self, action_pdf):
        possible_actions = list(action_pdf.keys())
        norm = sum(v for v in action_pdf.values())
        action_likelihoods = [action_pdf[a]/norm for a in possible_actions]
        action = self.rng_instance.choice(possible_actions, p=action_likelihoods)
        obs, reward, done, info = super().step(action)
        obs['prev_action'] = action
        return obs, reward, done, info

    @property
    def conditions(self):
        return self.conditions_w_seed

    def conditioned_reset(self, selected_condition):
        cond, seed = selected_condition
        self.rng_instance = np.random.RandomState(seed)
        return super().conditioned_reset(cond)
