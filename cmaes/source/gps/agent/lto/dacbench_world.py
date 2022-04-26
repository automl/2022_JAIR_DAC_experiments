import numpy as np
from collections import deque
from cma.evolution_strategy import CMAEvolutionStrategy, CMAOptions
from cma import bbobbenchmarks as bn
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_PS, CUR_SIGMA, PAST_LOC_DELTAS, PAST_SIGMA
from dacbench.benchmarks import CMAESBenchmark
import threading
import concurrent.futures

def _norm(x): return np.sqrt(np.sum(np.square(x)))
class DACBenchWorld(object):
    def __init__(self, instance_set_path, instance_id, seed):
        self.benchmark = CMAESBenchmark()
        self.benchmark.config["instance_set_path"] = instance_set_path
        self.seed = seed
        self.instance_id = instance_id


    def run(self, batch_size="all", ltorun=False):
        """Initiates the first time step"""

        self.environment.reset()
        self.environment.use_next_instance(instance_id=self.instance_id)
        self.environment.history.clear()
        self.environment.past_obj_vals.clear()
        self.environment.past_sigma.clear()
        self.instance = self.environment.instance
        self.environment.cur_loc = self.instance[3]
        self.environment.dim = self.instance[1]
        self.environment.init_sigma = self.instance[2]
        self.environment.cur_sigma = [self.environment.init_sigma]
        self.environment.fcn = bn.instantiate(self.instance[0])[0]

        self.environment.func_values = []
        self.environment.f_vals = deque(maxlen=self.environment.popsize)
        self.environment.es = CMAEvolutionStrategy(
            self.environment.cur_loc,
            self.environment.init_sigma,
            {"popsize": self.environment.popsize, "bounds": self.environment.bounds, "seed": self.seed},
        )
        self.es = self.environment.es
        self.func_values = self.environment.func_values
        self.environment.solutions, self.environment.func_values = self.environment.es.ask_and_eval(self.environment.fcn)
        self.environment.fbest = self.environment.func_values[np.argmin(self.environment.func_values)]
        self.fbest = self.environment.fbest
        self.environment.f_difference = np.abs(
            np.amax(self.environment.func_values) - self.environment.cur_obj_val
        ) / float(self.environment.cur_obj_val)
        self.environment.velocity = np.abs(np.amin(self.environment.func_values) - self.environment.cur_obj_val) / float(
            self.environment.cur_obj_val
        )
        self.environment.es.mean_old = self.environment.es.mean
        self.environment.history.append([self.environment.f_difference, self.environment.velocity])

    # action is of shape (dU,)
    def run_next(self, action):
        self.environment.step(action=action)
        self.fbest = self.environment.fbest

    def reset_world(self):
        self.environment = self.benchmark.get_environment()

    def get_state(self):
        env_state = self.environment.get_state(self.environment)
        state = {CUR_LOC: env_state["current_loc"],
                 PAST_OBJ_VAL_DELTAS: env_state["past_deltas"] + 1e-4,
                 CUR_PS: env_state["current_ps"] + 1e-4,
                 CUR_SIGMA: env_state["current_sigma"],
                 PAST_LOC_DELTAS: env_state["history_deltas"] + 1e-4,
                 PAST_SIGMA: env_state["past_sigma_deltas"] + 1e-4
                }
        return state

