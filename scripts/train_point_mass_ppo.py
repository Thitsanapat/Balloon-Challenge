"""Train a fast route policy before transferring it to the 6-DoF agent."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from BalloonPoppingGymEnv.training.point_mass_env import PointMassBalloonEnv


def expert_dataset(field, replay, model=None):
    payload = json.loads(Path(replay).read_text(encoding="utf-8"))
    records = payload["trajectories"] if isinstance(payload, dict) else payload
    times = np.asarray([record["time"] for record in records])
    positions = np.asarray([
        [np.nan if value is None else value for value in record["rocket_states"][:3]]
        for record in records
    ])
    env = PointMassBalloonEnv(field)
    observation, _ = env.reset()
    observations, actions = [], []
    while env.step_index < env.burnout_step:
        next_step = min(env.step_index + env.action_repeat, env.end_step)
        replay_index = int(np.argmin(np.abs(times - next_step * env.field_dt)))
        target = positions[replay_index]
        duration = (next_step - env.step_index) * env.field_dt
        acceleration = 2. * (
            target - env.position - env.velocity * duration
        ) / duration**2
        thrust = acceleration + np.array([0., 0., 9.80665])
        magnitude = np.linalg.norm(thrust)
        direction = thrust / max(magnitude, 1e-9)
        burn_fraction = ((env.step_index - env.launch_step)
                         / (env.burnout_step - env.launch_step))
        available = env.thrust_accel * (.88 + .26 * burn_fraction)
        action = np.r_[
            direction[:2] / .95,
            2. * np.clip(magnitude / available, 0., 1.) - 1.,
        ]
        action = np.clip(action, -1., 1.).astype(np.float32)
        observations.append(observation)
        actions.append(action)
        executed = action
        if model is not None:
            executed, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(executed)
        if terminated or truncated:
            break
    return np.asarray(observations), np.asarray(actions)


def behavior_clone(model, observations, actions, epochs):
    device = model.device
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    target = torch.as_tensor(actions, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    for epoch in range(int(epochs)):
        order = torch.randperm(len(obs), device=device)
        running = 0.
        for start in range(0, len(obs), 128):
            index = order[start:start + 128]
            features = model.policy.extract_features(obs[index])
            latent = model.policy.mlp_extractor.forward_actor(features)
            mean = model.policy.action_net(latent)
            loss = torch.mean((mean - target[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss) * len(index)
        if epoch in {0, int(epochs) - 1} or (epoch + 1) % 100 == 0:
            print(f"bc epoch={epoch + 1} mse={running / len(obs):.6f}")


def evaluate(model, field, reference_replay=None, residual_accel=3., bonus_targets=(),
             residual_start=float("-inf"), residual_end=float("inf")):
    env = PointMassBalloonEnv(
        field, reference_replay=reference_replay,
        residual_accel=residual_accel, bonus_targets=bonus_targets,
    )
    env.residual_start, env.residual_end = residual_start, residual_end
    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
    print(f"surrogate deterministic score={info['popped_count']} "
          f"indices={info['popped_indices']}")
    return info["popped_count"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("field", type=Path)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--subprocess", action="store_true",
                        help="Run environments in separate CPU processes")
    parser.add_argument("--expert-replay", type=Path)
    parser.add_argument("--bc-epochs", type=int, default=1000)
    parser.add_argument("--dagger-iterations", type=int, default=8)
    parser.add_argument("--residual-reference", type=Path)
    parser.add_argument("--residual-accel", type=float, default=3.)
    parser.add_argument("--bonus-targets", nargs="*", type=int, default=[])
    parser.add_argument("--residual-start", type=float, default=float("-inf"))
    parser.add_argument("--residual-end", type=float, default=float("inf"))
    parser.add_argument("--output", type=Path,
                        default=Path("BalloonPoppingGymEnv/evaluation/results/point_mass_ppo"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--log-std", type=float)
    parser.add_argument("--zero-policy", action="store_true")
    args = parser.parse_args()
    def make_training_env():
        training_env = PointMassBalloonEnv(
            args.field, reference_replay=args.residual_reference,
            residual_accel=args.residual_accel,
            bonus_targets=args.bonus_targets,
        )
        training_env.residual_start = args.residual_start
        training_env.residual_end = args.residual_end
        return training_env

    env = make_vec_env(
        make_training_env,
        n_envs=args.envs,
        seed=304729,
        vec_env_cls=SubprocVecEnv if args.subprocess else None,
    )
    if args.load is not None:
        model = PPO.load(args.load, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", env, learning_rate=1e-4, n_steps=1024,
            batch_size=512, n_epochs=10, gamma=.995, gae_lambda=.95,
            ent_coef=.01, verbose=1, device="cpu",
            policy_kwargs={"net_arch": [256, 256]},
        )
    if args.log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(args.log_std)
    if args.zero_policy:
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    if args.expert_replay is not None:
        observations, actions = expert_dataset(args.field, args.expert_replay)
        print(f"expert transitions={len(observations)}")
        behavior_clone(model, observations, actions, args.bc_epochs)
        evaluate(model, args.field, args.residual_reference,
                 args.residual_accel, args.bonus_targets,
                 args.residual_start, args.residual_end)
        for iteration in range(args.dagger_iterations):
            recovery_observations, recovery_actions = expert_dataset(
                args.field, args.expert_replay, model=model,
            )
            observations = np.r_[observations, recovery_observations]
            actions = np.r_[actions, recovery_actions]
            print(f"dagger iteration={iteration + 1} transitions={len(observations)}")
            behavior_clone(model, observations, actions, max(100, args.bc_epochs // 5))
            evaluate(model, args.field, args.residual_reference,
                     args.residual_accel, args.bonus_targets,
                     args.residual_start, args.residual_end)
    if args.steps > 0:
        model.learn(total_timesteps=args.steps, progress_bar=False)
        evaluate(model, args.field, args.residual_reference,
                 args.residual_accel, args.bonus_targets,
                 args.residual_start, args.residual_end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
