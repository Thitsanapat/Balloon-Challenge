"""Create a reproducible evaluation config from a trained linear ranker.

The learned numeric parameters are copied into YAML so the flight agent has no
runtime model/data-file dependency.  This tool never changes a scenario seed,
the simulator, or the submission packer.
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent-module', type=str, required=True)
    parser.add_argument('--rank-bonus', type=float, required=True)
    args = parser.parse_args()
    if not 0. <= args.rank_bonus <= .49:
        parser.error('--rank-bonus must be in [0, .49]')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    model = json.loads(args.model.read_text(encoding='utf-8'))
    required = ('coefficients', 'feature_mean', 'feature_scale')
    if any(key not in model for key in required):
        raise ValueError('Model is missing compact linear rank parameters')
    if not all(len(model[key]) == 16 for key in required):
        raise ValueError('Learned-beam ranker requires exactly 16 features')
    config = yaml.safe_load(args.template.read_text(encoding='utf-8-sig'))
    config['agent_module_path'] = args.agent_module
    config['agent_class_name'] = 'LearnedBeamSubmissionAgent'
    config['agent_name'] = 'Observation-only learned root-beam ranker'
    config.setdefault('agent_kwargs', {}).update(
        rank_coefficients=model['coefficients'],
        rank_feature_mean=model['feature_mean'],
        rank_feature_scale=model['feature_scale'],
        rank_bonus=float(args.rank_bonus),
    )
    config['leaderboard_submission'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
