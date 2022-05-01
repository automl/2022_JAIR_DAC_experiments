import argparse
import sys
import numpy as np

from dac.policies import SGDStaticPolicy, RMSpropPolicy, MomentumPolicy

from dacbench.benchmarks import SGDBenchmark


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', choices=['rmsprop', 'momentum'], type=str.lower, required=True,
                        help='The optimizer we control learning rate for in meta-test')
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], type=str.lower, required=True,
                        help='The dataset we train on in meta-test, i.e., the inner task')
    parser.add_argument('--baseline', type=float,
                        help='Test a specified constant learning rate (baseline)')
    parser.add_argument('--policy', type=str,
                        help='Path to .npy file containing learned parameters (as stored by run_meta_training)')
    parser.add_argument('--eval_frequency', type=float, default=1.5,
                        help='Factor controlling frequency of full training set evaluations')
    args = parser.parse_args()

    policy = None
    if args.baseline and not args.policy:
        print("> Testing baseline with constant learning rate (lr: {}) for {} on {}".format(
            args.baseline, args.optimizer, args.dataset))
        policy = SGDStaticPolicy(lr=args.baseline)
    elif args.policy and not args.baseline:
        print("> Testing learned policy for {} on {}".format(args.optimizer, args.dataset))
        config = np.load(args.policy)
        print("> Loaded learned parameters stored at {}".format(args.policy))
        if args.optimizer == "rmsprop":
            policy = RMSpropPolicy()
        elif args.optimizer == "momentum":
            policy = MomentumPolicy()
        policy.reconfigure(config)
    else:
        sys.exit(">>> FATAL ERROR: Exactly one of --baseline and --policy options must be specified")

    print("> Succesfully loaded policy to be tested: {}".format(policy))

    # create meta-test environment
    kwargs = {
        "instance_set_path": "../instance_sets/sgd/sgd_test_{}_{}.csv".format(args.optimizer, args.dataset),
        "optimizer": args.optimizer,
        "cd_paper_reconstruction": True,
        "cd_bias_correction": True,
        "training_batch_size": 512,
        "train_validation_ratio": 1.0,
        "features": [
            "predictiveChangeVarDiscountedAverage",
            "predictiveChangeVarUncertainty",
            "lossVarDiscountedAverage",
            "lossVarUncertainty",
            "currentLR",
            "trainingLoss",
            "step"
        ]
    }

    bench = SGDBenchmark(**kwargs)
    env = bench.get_environment()
    print("> Loaded meta-test environment")
    print("> Start meta-test run")
    obs = env.reset()
    policy.reset()
    step_count = 0
    print("optimization steps, full training error")
    print("{}, {}".format(step_count, env.get_full_training_loss()))
    count = 1
    next_log_count = np.ceil(pow(args.eval_frequency, count))

    done = False
    while not done and not obs.get("crashed", 0):
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        step_count += 1
        if step_count >= next_log_count:
            print("{}, {}".format(step_count, env.get_full_training_loss()))
            count += 1
            next_log_count = np.ceil(pow(args.eval_frequency, count))

    print("{}, {}".format(step_count, env.get_full_training_loss()))
    print("> Meta-test run ended")
