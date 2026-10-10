"""Single-worker mixed-scenario pilot; never resumes incompatible old policies.

Mixture weights concern episode sampling, not score weighting. Only released
official parameter files are selected. The simulator and official reward are
unchanged; shaped training returns are logged separately from actual pops.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train_official_residual_ppo import hashes, ROOT, RunControl, write_json
from BalloonPoppingGymEnv.training.multiscenario_residual_env import (
    MultiScenarioResidualEnv, DEFAULT_SCENARIOS, DEFAULT_WEIGHTS,
    SEED_POOLS, validate_mixture,
)
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import FEATURE_SCHEMA


def source_hashes():
    result = {}
    for scenario in DEFAULT_SCENARIOS:
        result.update(hashes(scenario))
    paths = [
        'scripts/train_multiscenario_residual_ppo.py',
        'BalloonPoppingGymEnv/agents/multiscenario_residual_agent.py',
        'BalloonPoppingGymEnv/agents/scenario4_joint_refresh_agent.py',
        'BalloonPoppingGymEnv/agents/constrained_beam_agent.py',
        'BalloonPoppingGymEnv/training/multiscenario_residual_env.py',
        'BalloonPoppingGymEnv/envs/balloon_world.py',
        'BalloonPoppingGymEnv/evaluation/evaluate.py',
    ]
    for name in paths:
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--scenarios', nargs='+', type=int, default=list(DEFAULT_SCENARIOS))
    p.add_argument('--weights', nargs='+', type=float, default=list(DEFAULT_WEIGHTS))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=20_000)
    p.add_argument('--hours', type=float, default=2.)
    p.add_argument('--seed', type=int, default=261004)
    p.add_argument('--n-steps', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--checkpoint-every', type=int, default=512)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        scenarios, weights = validate_mixture(args.scenarios, args.weights)
    except ValueError as error:
        p.error(str(error))
    if (min(args.steps, args.n_steps, args.batch_size, args.checkpoint_every) < 2
            or not np.isfinite(args.hours) or args.hours <= 0
            or args.batch_size > args.n_steps or args.n_steps % args.batch_size):
        p.error('Positive limits and batch size dividing rollout length required')
    if args.device == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or torch.cuda.device_count() != 1:
            p.error('Exactly one explicitly visible GPU required')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    started = time.monotonic()
    fingerprints = source_hashes()
    import stable_baselines3, gymnasium
    manifest = dict(
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        mixture=dict(scenarios=scenarios, probabilities=weights.tolist()),
        feature_schema=FEATURE_SCHEMA, seed_pools=SEED_POOLS,
        source_sha256=fingerprints, torch=torch.__version__, numpy=np.__version__,
        sb3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''), workers=1,
        policy=dict(layers=[256, 256], gamma=.995, learning_rate=5e-5, target_kl=.015),
        controller='full-route joint refresh, zero-residual bypass, active target first',
        action='three bounded acceleration corrections, 2 m/s^2/component, 10 physics steps/decision',
        observation='112 observation/controller/public-parameter features; no scenario ID, seed, private state or future',
        evaluation_gate='No promotion without fresh paired validation/holdout and no official truncations',
        budget='soft wall limit checked between decisions; simulator reset can delay stop',
        uploaded=False)
    write_json(directory/'manifest.json', manifest)
    callback = RunControl(directory, started, args.hours, args.checkpoint_every,
                          fingerprints, scenario='mixed', source_hash_provider=source_hashes)
    env = None
    try:
        env = DummyVecEnv([lambda: Monitor(MultiScenarioResidualEnv(
            scenarios=scenarios, weights=weights, split='train',
            reset_log=directory/'reset_failures.jsonl', episode_log=directory/'episode_seeds.jsonl'),
            filename=str(directory/'worker'),
            info_keywords=('popped_count', 'official_truncated', 'scenario'))])
        model = PPO('MlpPolicy', env, learning_rate=5e-5, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=5, gamma=.995,
                    gae_lambda=.95, ent_coef=.001, clip_range=.15, target_kl=.015,
                    max_grad_norm=.5, seed=args.seed, device=args.device, verbose=1,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-2.3))
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        model.save(directory/'initial_model')
        write_json(directory/'started.json', dict(pid=os.getpid(), started_unix=time.time(),
                                                  device=str(model.device)))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(callback, 'stop_requested', True))
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(directory/'final_model')
        write_json(directory/'progress.json', callback.status('stopped' if callback.stop_reason else 'completed'))
    except BaseException as error:
        write_json(directory/'error.json', dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
