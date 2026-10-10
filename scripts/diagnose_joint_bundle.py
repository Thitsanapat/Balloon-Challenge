"""Read-only semantic-source comparison for modular and bundled controllers."""

import ast
import argparse
import importlib.util
import inspect
from pathlib import Path
import textwrap

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


ROOT = Path(__file__).resolve().parents[1]


def syntax(function):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))),
                    include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=(
        ROOT / 'review/scenario4_pure_joint_candidate_v2.py'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'joint_bundle_diagnostic', args.bundle)
    bundled = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundled)
    seen = set()
    for cls in Scenario4PureJointRefreshAgent.__mro__:
        if not cls.__module__.startswith('BalloonPoppingGymEnv.agents.'):
            continue
        other = getattr(bundled, cls.__name__, None)
        if cls.__name__ == 'Scenario4WindProfileAgent':
            other = bundled.Scenario4SubmissionAgent
        if other is None:
            print('MISSING CLASS', cls.__name__)
            continue
        for name, method in vars(cls).items():
            if not inspect.isfunction(method):
                continue
            second = vars(other).get(name)
            if second is None or syntax(method) != syntax(second):
                print('METHOD DIFF', cls.__name__, name)
            for binding in method.__code__.co_names:
                value = method.__globals__.get(binding)
                if (not callable(value) or inspect.isclass(value) or
                        not getattr(value, '__module__', '').startswith(
                            'BalloonPoppingGymEnv.agents.')):
                    continue
                key = (value.__module__, binding)
                if key in seen:
                    continue
                seen.add(key)
                counterpart = method.__globals__.get(binding)
                if second is not None:
                    counterpart = second.__globals__.get(binding)
                if counterpart is None or syntax(value) != syntax(counterpart):
                    print('GLOBAL DIFF', cls.__name__, name, binding, value.__module__)


if __name__ == '__main__':
    main()
