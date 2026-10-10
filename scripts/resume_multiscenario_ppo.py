"""Bounded exact-model continuation; preserve the previously validated pilot.

This is a fresh episode stream, not a bitwise continuation of a partially
collected rollout. Model, optimizer and timestep counters are restored; the
environment and stochastic stream are explicitly reset and recorded.
"""
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train_multiscenario_residual_ppo import parser, source_hashes
from scripts.train_official_residual_ppo import ROOT, RunControl, write_json
from BalloonPoppingGymEnv.training.multiscenario_residual_env import MultiScenarioResidualEnv, validate_mixture
from BalloonPoppingGymEnv.training.resume_contract import validate_parent, sha256
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import FEATURE_SCHEMA, FEATURE_SIZE


def restore_policy(checkpoint, env, device, seed, n_steps, batch_size):
    model = PPO.load(checkpoint, env=env, device=device, force_reset=True)
    if (model.observation_space.shape != (FEATURE_SIZE,) or model.action_space.shape != (3,)
            or model.n_steps != n_steps or model.batch_size != batch_size):
        raise ValueError('Loaded model dimensions or rollout configuration differ')
    # Loading restores the old seed; explicitly reset the new episode/RNG stream.
    model.set_random_seed(seed)
    model.seed = seed
    return model


def main():
    p = parser()
    p.description = __doc__
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--resume-sha256', required=True)
    p.set_defaults(seed=261006)
    args = p.parse_args()
    scenarios, weights = validate_mixture(args.scenarios, args.weights)
    if (min(args.steps, args.n_steps, args.batch_size, args.checkpoint_every) < 2
            or not np.isfinite(args.hours) or args.hours <= 0
            or args.batch_size > args.n_steps or args.n_steps % args.batch_size):
        p.error('Positive limits and batch size dividing rollout length required')
    if args.device == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or torch.cuda.device_count() != 1:
            p.error('Exactly one explicitly visible GPU required')
    torch.set_num_threads(1)
    parent, provenance = validate_parent(args.resume, args.resume_sha256, ROOT,
                                         FEATURE_SCHEMA, scenarios, weights, args.n_steps, args.batch_size)
    if args.seed == parent['settings']['seed']:
        p.error('Use a distinct external episode seed for continuation')
    bundle = ROOT/'bundle_manifest.json'
    runtime = json.loads(bundle.read_text())['files'] if bundle.exists() else {}
    for name, expected in runtime.items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Frozen runtime differs: {name}')
    def fingerprints():
        result = source_hashes()
        for name in list(runtime)+['scripts/resume_multiscenario_ppo.py',
                                  'BalloonPoppingGymEnv/training/resume_contract.py']:
            result[name] = sha256(ROOT/name)
        return result
    fixed = fingerprints()
    parent_paths = [args.resume.resolve(), args.resume.resolve().parent/'manifest.json']
    parent_hashes = {str(path): sha256(path) for path in parent_paths}
    def guarded_hashes():
        result = fingerprints()
        # Parent files are immutable external inputs; don't silently continue if changed.
        if any(sha256(path) != value for path, value in parent_hashes.items()):
            result['parent_modified'] = 'true'
        return result
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    import stable_baselines3, gymnasium
    manifest = dict(parent)
    manifest.update(settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_sha256=fixed, resume=provenance, uploaded=False,
                    torch=torch.__version__, sb3=stable_baselines3.__version__,
                    numpy=np.__version__, gymnasium=gymnasium.__version__,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    continuation='Model and optimizer restored; new reset/episode RNG stream, partial rollout discarded',
                    evaluation_gate='Compare against the frozen parent on fresh paired seeds; no automatic promotion')
    write_json(directory/'manifest.json', manifest)
    started = time.monotonic()
    callback = RunControl(directory, started, args.hours, args.checkpoint_every, fixed,
                          scenario='mixed', source_hash_provider=guarded_hashes)
    env = None
    try:
        env = DummyVecEnv([lambda: Monitor(MultiScenarioResidualEnv(
            scenarios=scenarios, weights=weights, split='train',
            reset_log=directory/'reset_failures.jsonl', episode_log=directory/'episode_seeds.jsonl'),
            filename=str(directory/'worker'), info_keywords=('popped_count','official_truncated','scenario'))])
        model = restore_policy(args.resume, env, args.device, args.seed, args.n_steps, args.batch_size)
        # This is the restored parent, NOT a newly zero-initialized policy.
        model.save(directory/'initial_model')
        write_json(directory/'started.json', dict(pid=os.getpid(), started_unix=time.time(),
            device=str(model.device), parent_ppo_decisions=model.num_timesteps,
            parent_optimizer_epochs=model._n_updates, initial_model_is_restored_parent=True))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(callback, 'stop_requested', True))
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
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
