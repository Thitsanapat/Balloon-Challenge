"""One official flight compares packaged joint agent to its development source."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.build_scenario4_joint_submission_agent import build_source


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
ABLATION = RESULTS / 'sensor_coupled_gnc_dev_20261009'
OUTPUT = RESULTS / 'pure_joint_standalone_parity_v2_20261009'


def run():
    deadline = time.monotonic() + 3 * 3600
    while not (ABLATION / 'summary.json').exists():
        if (ABLATION / 'failed.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Ablation did not finish; parity not started')
        time.sleep(15)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = ROOT / 'review/scenario4_pure_joint_candidate_v2.py'
    if source.read_text(encoding='utf-8') != build_source():
        raise RuntimeError('Standalone does not reproduce from its builder')
    OUTPUT.mkdir()
    config = yaml.safe_load((ABLATION / '2_joint.yaml').read_text())
    config['agent_module_path'] = str(source.relative_to(ROOT)).replace('\\', '/')
    config['agent_name'] = 'Scenario 4 joint refresh standalone parity only'
    config_path = OUTPUT / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    report = OUTPUT / 'flight.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'), str(ROOT))))
    try:
        with (OUTPUT / 'flight.log').open('x', encoding='utf-8') as log:
            subprocess.run((sys.executable, '-m', 'scripts.evaluate_guidance',
                            str(config_path), '--seeds', '2', '--report', str(report)),
                           env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        reference = json.loads((ABLATION / '2_joint.json').read_text())[0]
        candidate = json.loads(report.read_text())[0]
        fields = ('score', 'final_time', 'terminated', 'truncated', 'pop_events',
                  'target_events', 'route_events', 'diagnostics', 'navigation_error',
                  'closest_true_distances', 'guidance_telemetry')
        checks = {field: reference[field] == candidate[field] for field in fields}
        checks.update(source_unchanged=candidate['agent_file_unchanged_during_run'],
                      dependencies_unchanged=candidate['agent_sources_unchanged_during_run'])
        summary = dict(seed=2, score=candidate['score'], checks=checks,
                       passed=all(checks.values()), live_agent_changed=False,
                       scope='One-seed behavior parity, not new generalization evidence or lockfile-environment parity')
        (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        (OUTPUT / 'failed.json').write_text(json.dumps(dict(error=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    run()
