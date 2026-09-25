"""Train and evaluate three observation-only high-level route selectors.

Models choose a rank among currently released target candidates. The inherited
chain controller produces all commands, preserves its physical checks, and
controls the rocket. Training artifacts may contain learned weights but never
stored balloon fields, target IDs, command sequences, or seed lookup tables.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.route_selector_agent import SELECTOR_FEATURES
from BalloonPoppingGymEnv.training.route_selector_env import OfficialSelectorEnv


def fresh_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def random_collection(args):
    output = fresh_path(args.output)
    env = OfficialSelectorEnv(selector_candidates=args.candidates,
                              decision_steps=args.decision_steps,
                              record_queries=True,
                              reset_log=output.with_suffix('.reset_failures.jsonl'))
    rng = np.random.default_rng(args.seed)
    features, masks, actions, returns = [], [], [], []
    health = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=int(rng.integers(2**31-1)))
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.action_masks()
                action = int(rng.choice(np.flatnonzero(mask)))
                observation, _, terminated, truncated, info = env.step(action)
            score = int(info['popped_count'])
            for query in env.queries:
                if not query['root_selected']:
                    continue
                features.append(query['features'])
                masks.append(query['mask'])
                actions.append(query['action'])
                returns.append(score)
            health.append(dict(episode=episode, score=score, truncated=bool(truncated),
                               simulation_time=info['simulation_time'],
                               root_queries=sum(q['root_selected'] for q in env.queries)))
            print(json.dumps(health[-1]), flush=True)
    finally:
        env.close()
    if not features:
        raise RuntimeError('Collection contained no released-target queries')
    np.savez_compressed(output, features=np.asarray(features, dtype=np.float32),
                        masks=np.asarray(masks, dtype=bool), actions=np.asarray(actions),
                        returns=np.asarray(returns, dtype=np.float32),
                        health=np.asarray(json.dumps(health)))
    print(json.dumps({'dataset': str(output), 'queries': len(features),
                      'episodes': len(health), 'mean_score': float(np.mean([x['score'] for x in health]))}))


def merge_collections(args):
    """Concatenate separately completed random-policy datasets losslessly."""
    if not args.datasets:
        raise ValueError('At least one --datasets input is required')
    arrays, health = [], []
    for path in args.datasets:
        source = np.load(path, allow_pickle=False)
        required = ('features', 'masks', 'actions', 'returns', 'health')
        if any(key not in source for key in required):
            raise ValueError(f'{path} is not a selector collection')
        arrays.append({key: source[key] for key in required[:-1]})
        health.extend(json.loads(str(source['health'].item())))
    shapes = {entry['features'].shape[1:] for entry in arrays}
    if len(shapes) != 1:
        raise ValueError('Cannot merge different feature shapes')
    output = fresh_path(args.output)
    np.savez_compressed(output,
                        features=np.concatenate([x['features'] for x in arrays]),
                        masks=np.concatenate([x['masks'] for x in arrays]),
                        actions=np.concatenate([x['actions'] for x in arrays]),
                        returns=np.concatenate([x['returns'] for x in arrays]),
                        health=np.asarray(json.dumps(health)))
    print(json.dumps({'dataset': str(output), 'episodes': len(health),
                      'queries': int(sum(len(x['actions']) for x in arrays))}))


def train_xgb(args):
    import xgboost as xgb
    source = np.load(args.dataset, allow_pickle=False)
    features, masks = source['features'], source['masks']
    actions, returns = source['actions'].astype(int), source['returns']
    candidates = masks.shape[1]
    rows, labels, qids = [], [], []
    for query, (matrix, mask, action, outcome) in enumerate(zip(features, masks, actions, returns)):
        local = matrix[:candidates*SELECTOR_FEATURES].reshape(candidates, SELECTOR_FEATURES)
        global_features = matrix[candidates*SELECTOR_FEATURES:]
        for rank in np.flatnonzero(mask):
            rows.append(np.r_[local[rank], global_features])
            # Only the selected candidate received the observed episode return.
            # This is contextual bandit feedback, not a counterfactual label.
            labels.append(float(outcome) if rank == action else 0.)
            qids.append(query)
    if len(set(qids)) < 2 or not np.any(labels):
        raise RuntimeError('Dataset needs multiple nonzero-return query groups')
    order = np.argsort(qids, kind='stable')
    groups = np.bincount(np.asarray(qids, dtype=int)[order])
    matrix = xgb.DMatrix(np.asarray(rows)[order], label=np.asarray(labels)[order])
    matrix.set_group(groups)
    model = xgb.train({'objective': 'rank:ndcg', 'tree_method': 'hist', 'device': 'cpu',
                       'max_depth': args.depth, 'eta': args.learning_rate,
                       'subsample': .85, 'colsample_bytree': .9,
                       'lambdarank_pair_method': 'mean',
                       'lambdarank_num_pair_per_sample': 4, 'seed': args.seed},
                      matrix, num_boost_round=args.trees)
    output = fresh_path(args.output)
    model.save_model(output)
    print(json.dumps({'model': str(output), 'queries': len(set(qids)), 'rows': len(rows),
                      'positive_rows': int(np.count_nonzero(labels))}))


def train_linear(args):
    """Fit a compact ridge score from selected contextual-bandit examples.

    Each chosen row receives only its completed episode's actual pop count.
    This deliberately avoids assigning fabricated zero counterfactual outcomes
    to candidates that the random policy did not select.  The output is numeric
    parameters suitable for a self-contained observation-only agent config.
    """
    source = np.load(args.dataset, allow_pickle=False)
    features, masks = source['features'], source['masks']
    actions, returns = source['actions'].astype(int), source['returns']
    candidates = masks.shape[1]
    rows = []
    for matrix, mask, action in zip(features, masks, actions):
        if not (0 <= action < candidates and mask[action]):
            continue
        local = matrix[:candidates*SELECTOR_FEATURES].reshape(
            candidates, SELECTOR_FEATURES)
        rows.append(np.r_[local[action], matrix[candidates*SELECTOR_FEATURES:]])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(returns), SELECTOR_FEATURES+4):
        raise RuntimeError('Selected contextual-bandit rows are malformed')
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-4)
    standardized = (matrix-mean)/scale
    design = np.c_[standardized, np.ones(len(standardized))]
    penalty = np.diag(np.r_[np.full(standardized.shape[1], args.ridge), 0.])
    solution = np.linalg.solve(design.T@design+penalty, design.T@returns)
    output = fresh_path(args.output)
    report = {
        'architecture': 'selected-contextual-bandit-ridge-v1',
        'examples': int(len(matrix)), 'ridge': float(args.ridge),
        'coefficients': solution[:-1].tolist(), 'intercept': float(solution[-1]),
        'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
        'training_rmse': float(np.sqrt(np.mean((design@solution-returns)**2))),
        'warning': ('Observed terminal scores label selected actions only; this '
                    'small pilot is not counterfactual policy evaluation.'),
    }
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(output), 'examples': len(matrix),
                      'training_rmse': report['training_rmse']}))


def xgb_policy(path, candidates):
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(path)
    def choose(observation, mask):
        local = observation[:candidates*SELECTOR_FEATURES].reshape(candidates, SELECTOR_FEATURES)
        global_features = observation[candidates*SELECTOR_FEATURES:]
        rows = np.c_[local, np.broadcast_to(global_features, (candidates, len(global_features)))]
        scores = model.predict(xgb.DMatrix(rows))
        scores = np.where(mask, scores, -np.inf)
        return int(np.argmax(scores))
    return choose


class AttentionPolicy:  # Imported only for the attention pilot.
    def __init__(self, candidates, device):
        import torch
        from torch import nn
        self.torch, self.candidates = torch, candidates
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.key = nn.Sequential(nn.Linear(SELECTOR_FEATURES+4, 64), nn.Tanh(),
                                         nn.Linear(64, 32))
                self.query = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 32))
                self.bias = nn.Sequential(nn.Linear(SELECTOR_FEATURES+4, 32), nn.Tanh(),
                                          nn.Linear(32, 1))
            def forward(self, observation):
                batch = observation.shape[0]
                targets = observation[:, :candidates*SELECTOR_FEATURES].reshape(
                    batch, candidates, SELECTOR_FEATURES)
                global_features = observation[:, candidates*SELECTOR_FEATURES:]
                joined = torch.cat((targets, global_features[:, None, :].expand(-1, candidates, -1)), dim=-1)
                return (self.key(joined)*self.query(global_features)[:, None, :]).sum(-1)/np.sqrt(32.) + self.bias(joined).squeeze(-1)
        self.net = Net().to(device)

    def logits(self, observation):
        return self.net(observation)


def attention_policy(path, candidates, device):
    import torch
    saved = torch.load(path, map_location=device, weights_only=True)
    policy = AttentionPolicy(candidates, device)
    policy.net.load_state_dict(saved['state_dict'])
    policy.net.eval()
    def choose(observation, mask):
        with torch.no_grad():
            logits = policy.logits(torch.as_tensor(observation, dtype=torch.float32, device=device)[None])[0]
            logits[~torch.as_tensor(mask, device=device)] = -torch.inf
            return int(torch.argmax(logits).item())
    return choose


def attention_train(args):
    import torch
    from torch.distributions import Categorical
    device = torch.device(args.device)
    output = fresh_path(args.output)
    env = OfficialSelectorEnv(selector_candidates=args.candidates,
                              decision_steps=args.decision_steps,
                              reset_log=output.with_suffix('.reset_failures.jsonl'))
    policy = AttentionPolicy(args.candidates, device)
    optimizer = torch.optim.Adam(policy.net.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    history, baseline = [], 0.
    started = time.monotonic()
    try:
        for episode in range(args.episodes):
            if time.monotonic()-started >= args.hours*3600:
                break
            observation, _ = env.reset(seed=int(rng.integers(2**31-1)))
            logps, entropies, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.action_masks()
                tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
                logits = policy.logits(tensor)[0]
                logits = logits.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                observation, reward, terminated, truncated, info = env.step(int(action.item()))
                logps.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                rewards.append(reward)
            returns, value = [], 0.
            for reward in reversed(rewards):
                value = reward + args.gamma*value
                returns.append(value)
            returns = torch.as_tensor(returns[::-1], dtype=torch.float32, device=device)
            episode_return = float(returns[0]) if len(returns) else 0.
            baseline = .9*baseline+.1*episode_return if history else episode_return
            advantage = returns-baseline
            advantage = advantage/(advantage.std(unbiased=False)+1e-6)
            loss = -(torch.stack(logps)*advantage).mean()-.001*torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.)
            optimizer.step()
            item = dict(episode=episode, score=int(info['popped_count']), truncated=bool(truncated),
                        steps=len(rewards), return_value=episode_return, loss=float(loss.item()))
            history.append(item)
            print(json.dumps(item), flush=True)
    finally:
        env.close()
    torch.save({'architecture': 'attention_selector_v1', 'candidates': args.candidates,
                'state_dict': policy.net.state_dict()}, output)
    summary = output.with_suffix('.json')
    summary.write_text(json.dumps({'model': str(output), 'history': history,
                                   'wall_seconds': time.monotonic()-started}, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(output), 'episodes': len(history), 'summary': str(summary)}))


def ppo_train(args):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    output = fresh_path(args.output)
    class HealthStop(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started, self.consecutive_truncations = time.monotonic(), 0
        def _on_step(self):
            for info in self.locals.get('infos', []):
                self.consecutive_truncations = self.consecutive_truncations+1 if info.get('official_truncated') else 0
            return (time.monotonic()-self.started < args.hours*3600 and self.consecutive_truncations < 3)
    env = OfficialSelectorEnv(selector_candidates=args.candidates,
                              decision_steps=args.decision_steps,
                              reset_log=output.with_suffix('.reset_failures.jsonl'))
    try:
        model = MaskablePPO('MlpPolicy', env, seed=args.seed, device=args.device,
                            n_steps=64, batch_size=64, n_epochs=3, gamma=args.gamma,
                            learning_rate=args.learning_rate, policy_kwargs={'net_arch': [128, 128]}, verbose=1)
        model.learn(total_timesteps=args.steps, callback=HealthStop())
        model.save(str(output))
    finally:
        env.close()
    print(json.dumps({'model': str(output), 'requested_steps': args.steps}))


def evaluate(args):
    if args.kind == 'baseline':
        choose = lambda observation, mask: 0
    elif args.model_file.suffix == '.json':
        choose = xgb_policy(args.model_file, args.candidates)
    elif args.kind == 'attention':
        choose = attention_policy(args.model_file, args.candidates, args.device)
    else:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model_file, device=args.device)
        choose = lambda observation, mask: int(model.predict(observation, deterministic=True, action_masks=mask)[0])
    output = fresh_path(args.output)
    env = OfficialSelectorEnv(selector_candidates=args.candidates, decision_steps=args.decision_steps,
                              reset_log=output.with_suffix('.reset_failures.jsonl'),
                              selector_enabled=args.kind != 'baseline')
    rng, rows = np.random.default_rng(args.seed), []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=int(rng.integers(2**31-1)))
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.action_masks()
                observation, _, terminated, truncated, info = env.step(choose(observation, mask))
            rows.append(dict(episode=episode, score=int(info['popped_count']),
                             truncated=bool(truncated), simulation_time=info['simulation_time']))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    output.write_text(json.dumps({'model': str(args.model_file), 'kind': args.kind,
                                  'scores': rows, 'mean_score': float(np.mean([x['score'] for x in rows]))}, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(output), 'mean_score': float(np.mean([x['score'] for x in rows]))}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'merge', 'xgb', 'linear', 'attention', 'ppo', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--datasets', type=Path, nargs='+')
    parser.add_argument('--model-file', type=Path)
    parser.add_argument('--kind', choices=('baseline', 'xgb', 'attention', 'ppo'), default='xgb')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--decision-steps', type=int, default=40)
    parser.add_argument('--trees', type=int, default=80)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--ridge', type=float, default=10.)
    parser.add_argument('--gamma', type=float, default=.995)
    parser.add_argument('--hours', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=240925)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.hours <= 0:
        parser.error('episodes, steps and hours must be positive')
    if args.command == 'collect':
        random_collection(args)
    elif args.command == 'merge':
        merge_collections(args)
    elif args.command == 'xgb':
        if args.dataset is None:
            parser.error('--dataset required for xgb')
        train_xgb(args)
    elif args.command == 'linear':
        if args.dataset is None:
            parser.error('--dataset required for linear')
        if args.ridge < 0:
            parser.error('--ridge must be nonnegative')
        train_linear(args)
    elif args.command == 'attention':
        attention_train(args)
    elif args.command == 'ppo':
        ppo_train(args)
    else:
        if args.kind != 'baseline' and args.model_file is None:
            parser.error('--model-file required for evaluate')
        evaluate(args)


if __name__ == '__main__':
    main()
