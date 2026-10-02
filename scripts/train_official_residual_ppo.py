"""Bounded, checkpointed PPO on fresh official scenario episodes.

Set CUDA_VISIBLE_DEVICES to ONE GPU UUID before starting this script, or use
--device cpu with CUDA_VISIBLE_DEVICES=''. No simulator/score modifications.
All steps reported as PPO steps are held-action decisions, not physics steps.
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from BalloonPoppingGymEnv.training.official_residual_env import OfficialResidualEnv
from BalloonPoppingGymEnv.training.run_health import TruncationGuard

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['scripts/train_official_residual_ppo.py',
           'BalloonPoppingGymEnv/training/run_health.py',
           'BalloonPoppingGymEnv/training/official_residual_env.py',
           'BalloonPoppingGymEnv/agents/residual_ppo_agent.py',
           'BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py']


def hashes(scenario=1):
    parameter_dir = 'BalloonPoppingGymEnv/envs/scenario_parameters'
    paths = SOURCES + [f'{parameter_dir}/scenario_{scenario}_{suffix}.yaml'
                       for suffix in ('parameters', 'given_parameters')]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def make_training_env(scenario, reset_log):
    return OfficialResidualEnv(scenario=scenario, reset_log=reset_log)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class RunControl(BaseCallback):
    def __init__(self, directory, started, max_hours, checkpoint_every, source_hashes,
                 truncation_limit=3, scenario=1):
        super().__init__()
        self.directory, self.started = directory, started
        self.max_seconds = max_hours*3600.
        self.checkpoint_every = checkpoint_every
        self.next_checkpoint = checkpoint_every
        self.last_status = 0.
        self.physics_steps = 0
        self.completed = []
        self.stop_requested = False
        self.stop_reason = None
        self.source_hashes = source_hashes
        self.scenario = scenario
        self.initial_steps = 0
        self.truncation_guard = TruncationGuard(truncation_limit)

    def _on_training_start(self):
        self.initial_steps = self.model.num_timesteps
        self.next_checkpoint = self.model.num_timesteps+self.checkpoint_every

    def status(self, state):
        elapsed = time.monotonic()-self.started
        completed = self.completed[-100:]
        return dict(status=state, pid=os.getpid(), ppo_decisions=int(self.num_timesteps),
            new_ppo_decisions=int(self.num_timesteps-self.initial_steps),
            controlled_physics_steps=self.physics_steps, warmup_steps_included=False,
            wall_seconds=elapsed, decisions_per_second=(self.num_timesteps-self.initial_steps)/max(elapsed, 1.),
            completed_episodes=len(self.completed), recent_episode_scores=completed,
            truncated_episodes=self.truncation_guard.total,
            mean_recent_score=float(np.mean([r['score'] for r in completed])) if completed else None,
            stop_reason=self.stop_reason, source_unchanged=hashes(self.scenario)==self.source_hashes,
            uploaded=False)

    def _on_step(self):
        unhealthy = False
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.physics_steps += int(info.get('physics_steps', 0))
            if done:
                row = dict(score=int(info['popped_count']), truncated=bool(info['official_truncated']),
                           ppo_decisions=int(self.num_timesteps),
                           episode_wall_seconds=info.get('episode_wall_seconds'),
                           simulation_time=info.get('simulation_time'))
                unhealthy = self.truncation_guard.observe(row['truncated']) or unhealthy
                self.completed.append(row)
                with (self.directory/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
        elapsed = time.monotonic()-self.started
        if self.num_timesteps >= self.next_checkpoint:
            self.model.save(self.directory/f'checkpoint_{self.num_timesteps}')
            self.next_checkpoint = self.num_timesteps+self.checkpoint_every
        if elapsed-self.last_status >= 30.:
            write_json(self.directory/'progress.json', self.status('running'))
            self.last_status = elapsed
        if self.stop_requested or (self.directory/'STOP').exists():
            self.stop_reason = 'requested_stop'
        elif unhealthy:
            self.stop_reason = 'consecutive_official_truncations'
        elif elapsed >= self.max_seconds:
            self.stop_reason = 'wall_time_budget'
        elif hashes(self.scenario) != self.source_hashes:
            self.stop_reason = 'source_changed'
        return self.stop_reason is None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--steps', type=int, default=2_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--checkpoint-every', type=int, default=10_000)
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=240926)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-consecutive-truncations', type=int, default=3)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.envs <= 8 or min(args.steps, args.n_steps, args.batch_size, args.checkpoint_every, args.max_consecutive_truncations) < 1 or args.hours <= 0:
        parser.error('Positive bounded settings required; at most eight workers')
    if args.device == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or torch.cuda.device_count() != 1:
            parser.error('GPU training requires exactly one explicitly visible GPU')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    started = time.monotonic()
    source_hashes = hashes(args.scenario)
    import stable_baselines3, gymnasium
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=source_hashes, torch=torch.__version__, sb3=stable_baselines3.__version__,
        numpy=np.__version__, gymnasium=gymnasium.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        observation='96 finite observation/controller-history features, no seed/private state/future trajectory',
        action='3 bounded residual acceleration components, 2 m/s^2 per component; held for 10 official steps',
        training=f'fresh unmodified official Scenario {args.scenario} per episode; randomized external seed',
        policy=dict(layers=[512, 512, 256], gamma=.995, learning_rate=5e-5, target_kl=.015),
        resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None)
    write_json(directory/'manifest.json', manifest)
    def factory(rank):
        def make():
            # Monitor logs training reward separately from official popped_count.
            return Monitor(make_training_env(args.scenario, directory/f'reset_failures_{rank}.jsonl'), filename=str(directory/f'worker_{rank}'),
                           info_keywords=('popped_count', 'official_truncated'))
        return make
    env = SubprocVecEnv([factory(i) for i in range(args.envs)], start_method='spawn')
    env.seed(args.seed)
    callback = RunControl(directory, started, args.hours, args.checkpoint_every,
                          source_hashes, args.max_consecutive_truncations,
                          scenario=args.scenario)
    def request_stop(signum, frame):
        callback.stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
            if model.n_steps != args.n_steps:
                raise ValueError('Resume n_steps must match the saved rollout configuration')
        else:
            model = PPO('MlpPolicy', env, learning_rate=5e-5, n_steps=args.n_steps,
                batch_size=args.batch_size, n_epochs=args.epochs, gamma=.995,
                gae_lambda=.95, ent_coef=.001, clip_range=.15, target_kl=.015,
                max_grad_norm=.5, seed=args.seed, device=args.device, verbose=1,
                policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), log_std_init=-1.6))
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
        # PPO.load restores its original seed; set the external episode stream
        # explicitly afterwards so the manifest's requested seed remains true.
        env.seed(args.seed)
        write_json(directory/'started.json', dict(pid=os.getpid(), started_unix=time.time(), device=str(model.device)))
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(directory/'final_model')
        write_json(directory/'progress.json', callback.status('stopped' if callback.stop_reason else 'completed'))
    except BaseException as exc:
        write_json(directory/'error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        try:
            env.close()
        except (BrokenPipeError, EOFError):
            # A crashed worker can close its pipe first. Reap only this vector
            # environment's children; do not mask the recorded original error.
            for process in env.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.)


if __name__ == '__main__':
    main()
