"""Generate one full balloon field for repeatable offline controller benchmarks."""

import argparse
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenario, _ = load_scenario_parameters(args.scenario)
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    env.reset(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        balloon_flights=env._balloon_flights,
        release_steps=env._balloon_release_at_step,
        scenario=np.array(args.scenario),
        seed=np.array(args.seed),
        time_step=np.array(scenario["simulation"]["time_step"]),
    )
    print(
        f"saved={args.output} shape={env._balloon_flights.shape} "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
