import sys
import os
import time
import socket
import json
import hashlib
import numpy as np
import exp_util as exp_util

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from smac.initial_design.random_configuration_design import RandomConfigurations

from cfg_util import vectorize_config, scale_vector

# GLOBAL variables (...) to be passed to target runner
GLOBAL = None


class Globals:

    def __init__(self, env, policy, param_scale, result_cache):
        self.env = env
        self.policy = policy
        self.param_scale = param_scale
        self.result_cache = result_cache
        self.n_evals_to_n_steps = [0]
        self.initial_costs = {}


# Target algorithm runner
def evaluate_cost(cfg, seed, instance, **kwargs):
    global GLOBAL

    # measure cost in steps
    cutoff = GLOBAL.env.conditions[int(instance)][3]
    GLOBAL.n_evals_to_n_steps.append(GLOBAL.n_evals_to_n_steps[-1] + cutoff)

    # Pre-processing on config
    cfg = vectorize_config(cfg)
    cfg = scale_vector(cfg, GLOBAL.param_scale)

    # Check result cache
    result_cache = GLOBAL.result_cache
    if result_cache is not None:
        key = hashlib.md5('{}+{}+{}'.format(cfg, seed, instance).encode()).hexdigest()
        value = result_cache.get(key)
        if value is not None:
            return value

    # Not in result cache, run target algorithm
    policy = GLOBAL.policy
    policy.reconfigure(cfg)
    policy.reset()
    env = GLOBAL.env
    condition = env.conditions[int(instance)]
    obs = env.conditioned_reset(condition)
    cutoff = condition[3]

    GLOBAL.initial_costs[instance] = GLOBAL.initial_costs.get(instance, env.env.get_full_training_loss())
    initial_loss = GLOBAL.initial_costs[instance]

    total_cost = 0
    done = False
    steps = 0
    while not done and not obs.get("crashed", 0):
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        # total_cost += -reward
        steps += 1

    # total_cost = total_cost / cutoff if (total_cost < 0 and not obs.get("crashed", 0)) else 0.0

    if not obs.get("crashed", 0):
        final_loss = env.env.get_full_training_loss()
        total_cost = min((np.log(final_loss) - np.log(initial_loss)) / cutoff, 0)

    # print('trial: {} (condition: {})'.format(total_reward, condition))
    print('[{}] {}: {}'.format(condition, policy.current_configuration(), total_cost))

    if result_cache is not None:
        # store result in cache
        result_cache.store(key, total_cost)

    return total_cost


def main(setup):
    global GLOBAL

    start_time = time.time()

    # set up solution quality tracer / output directories for results
    os.makedirs(setup['result_dir'], exist_ok=True)
    meta_info = {'start_time': start_time, 'host': socket.gethostname()}
    sqt_id = exp_util.store_result(setup['result_dir'], setup, meta=meta_info, overwrite=setup['overwrite'])
    smac_output_dir = os.path.join(setup['result_dir'], '{}_smac'.format(sqt_id))

    # set up Globals (required by target runner)
    env = setup['train_env']
    policy = env.policy_space()
    param_scale = setup['param_scale']
    cfg = policy.current_configuration()
    result_cache = None
    if setup.get("cache_evaluations", False):
        result_cache_path = os.path.join(setup['result_dir'], '{}.cache'.format(sqt_id))
        result_cache = exp_util.ResultCache(result_cache_path)
    GLOBAL = Globals(env, policy, param_scale, result_cache)

    # set up SMAC

    # Build configuration space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    # weights = [UniformFloatHyperparameter("w{}".format(i), -10, 10) for i in range(5)]
    params = []
    x_min, x_max = param_scale.domain()
    for i in range(len(cfg)):
        params.append(
            UniformFloatHyperparameter("p_{}".format(i), x_min, x_max))
    cs.add_hyperparameters(params)

    # Build Scenario
    scenario_dict = {}
    scenario_dict["run_obj"] = "quality"  # we optimize quality (alternatively runtime)
    scenario_dict["runcount-limit"] = setup['trials_train_limit']
    scenario_dict["cs"] = cs
    scenario_dict["deterministic"] = setup['deterministic']
    scenario_dict["instances"] = [[i] for i in range(len(env.conditions))]
    scenario_dict["output_dir"] = smac_output_dir
    scenario_dict["limit_resources"] = False
    scenario_dict["cost_for_crash"] = 1.0
    scenario_dict["abort_on_first_run_crash"] = False
    scenario = Scenario(scenario_dict)

    smac = SMAC4AC(scenario=scenario, rng=setup["seed"], tae_runner=evaluate_cost, initial_design=RandomConfigurations)

    # run smac (training)
    print('--- TRAINING ---')
    incumbent = smac.optimize()

    # validate incumbents & store SQT
    print('--- VALIDATION ---')
    val_env = setup['val_env']
    GLOBAL.env = val_env
    GLOBAL.policy = val_env.policy_space()
    tracer = exp_util.SQTRecorder(time_label='episodes_trained', verbose=True)
    traj_json = os.path.join(smac_output_dir, 'run_{}'.format(setup["seed"]), 'traj.json')
    with open(traj_json) as file:
        for i, line in enumerate(file):
            entry = json.loads(line)
            n_evals = entry['evaluations']
            if n_evals <= 0:
                continue
            n_steps = GLOBAL.n_evals_to_n_steps[n_evals]
            # store policy
            inc_i = vectorize_config(entry['incumbent'])
            checkpoint_file = os.path.join(setup['result_dir'], sqt_id, '{}.npy'.format(i))
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            np.save(checkpoint_file, inc_i)
            # evaluate the policy
            costs = np.empty((len(val_env.conditions),))
            for j, cond_j in enumerate(val_env.conditions):
                costs[j] = evaluate_cost(entry['incumbent'], 0, str(j))
            f_val = np.mean(costs)
            tracer.log(n_evals, steps_trained=n_steps, f_val=f_val, inc_id=i)
    sqt = tracer.produce()
    exp_util.store_result(setup['result_dir'], setup, meta_info, sqt, overwrite=True)


if __name__ == '__main__':
    config_file = sys.argv[1]
    vargs = sys.argv[2:]
    config_dict = exp_util.load_json_template(config_file, vargs)
    configuration = exp_util.parse_config_dict(config_dict)
    main(configuration)
