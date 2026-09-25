"""Export deterministic PPO actor + frozen planner as standalone NumPy source.

Only learned actor weights are embedded, not observations, seeds, replays or
environment objects. Full checkpoints stay available for training reproduction.
"""

import argparse
import ast
import hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]


def actor_arrays(model):
    import torch
    arrays = []
    for layer in model.policy.mlp_extractor.policy_net:
        if isinstance(layer, torch.nn.Linear):
            arrays.append((layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()))
        elif not isinstance(layer, torch.nn.Tanh):
            raise ValueError('Only Linear/Tanh actors are supported')
    layer = model.policy.action_net
    arrays.append((layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()))
    return arrays


def numpy_actor(arrays, features):
    x = np.asarray(features, dtype=np.float32)
    for index, (weight, bias) in enumerate(arrays):
        x = weight@x+bias
        if index < len(arrays)-1:
            x = np.tanh(x)
    return np.clip(x, -1., 1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen exported policy')
    model = PPO.load(args.checkpoint, device='cpu')
    arrays = actor_arrays(model)
    rng = np.random.default_rng(813)
    error = 0.
    for features in rng.uniform(-3., 3., (32, 96)).astype(np.float32):
        expected, _ = model.predict(features, deterministic=True)
        error = max(error, float(np.max(np.abs(expected-numpy_actor(arrays, features)))))
    if error > 2e-5:
        raise ValueError(f'Actor export mismatch: {error}')
    source = (ROOT/'BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py').read_text(encoding='utf-8')
    residual = ast.parse((ROOT/'BalloonPoppingGymEnv/agents/residual_ppo_agent.py').read_text(encoding='utf-8'))
    parts = [source]
    for node in residual.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, ast.Expr):
            continue
        parts.append(ast.unparse(node))
    parts.append('ACTOR_ARRAYS = [')
    for weight, bias in arrays:
        parts.append(f'(np.array({weight.tolist()!r}, dtype=np.float32), np.array({bias.tolist()!r}, dtype=np.float32)),')
    parts.append(']')
    parts.append('''class ExportedResidualAgent(ResidualPolicyAgent):
    def policy_action(self, features):
        x = np.asarray(features, dtype=np.float32)
        for index, (weight, bias) in enumerate(ACTOR_ARRAYS):
            x = weight @ x + bias
            if index < len(ACTOR_ARRAYS)-1:
                x = np.tanh(x)
        return np.clip(x, -1., 1.)
''')
    output = '\n\n'.join(parts)+'\n'
    compile(output, str(args.output), 'exec')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding='utf-8', newline='\n')
    print('checkpoint_sha256', hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    print('export_sha256', hashlib.sha256(output.encode()).hexdigest())
    print('actor_max_abs_error', error)
    print('output', args.output)


if __name__ == '__main__':
    main()
