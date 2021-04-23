import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents import dqn
from ray.rllib.agents.callbacks import DefaultCallbacks

from dacbench import benchmarks
from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper, ObservationWrapper
from pathlib import Path
import argparse


def make_benchmark(config):
    bench = getattr(benchmarks, config["benchmark"])()
    env = bench.get_benchmark(seed=config["seed"])
    if config["benchmark"] in ["SGDBenchmark", "CMAESBenchmark"]:
        env = ObservationWrapper(env)
    return env


parser = argparse.ArgumentParser(description="Run ray DQN for DACBench")
parser.add_argument("--outdir", type=str, default="output", help="Output directory")
parser.add_argument(
    "--benchmarks", nargs="+", type=str, default=None, help="Benchmarks to run PPO for"
)
parser.add_argument(
    "--timesteps", type=int, default=1000000, help="Number of timesteps to run"
)
parser.add_argument("--save_interval", type=int, default=100, help="Checkpointing interval")
parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="Seeds for evaluation",
)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--fd_port", type=int, default=55555)
args = parser.parse_args()

for b in args.benchmarks:
    for s in args.seeds:
        config = {"seed": s, "benchmark": b}
        if b == "FastDownwardBenchmark":
            config["port"] = args.fd_port
        register_env(f"{b}", lambda conf: make_benchmark(conf))
        ray.init()
        trainer = dqn.DQNTrainer(config={"num_gpus": 0,
                                        "env": f"{b}",
                                        "env_config": config,
                                        "framework": "tf" if not args.torch else "torch"})
        for i in range(args.timesteps):
            trainer.train()
            if i % args.save_interval == 0:
                trainer.save(args.outdir+f"/{b}_{s}")
        ray.shutdown()
