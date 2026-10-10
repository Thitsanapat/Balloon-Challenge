"""Final-night bounded development screen; never uploads or promotes a policy."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
from scripts.validate_sensor_coupled_gnc import read_flight, summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
OUTPUT = RESULTS / 'proactive_joint_dev_20261010'
REFERENCE = RESULTS / 'sensor_coupled_gnc_dev_20261009'
SEEDS = (0, 2)


def main():
    references = {seed: read_flight(REFERENCE / f'{seed}_joint.json') for seed in SEEDS}
    for row in references.values():
        for name, digest in row['agent_source_dependencies_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Reference dependency changed: {name}')
    registry = json.loads((ROOT / 'qualification/used_seeds.json').read_text())
    if any(str(seed) not in registry['seeds'] for seed in SEEDS):
        raise ValueError('Development seeds must already be recorded')
    OUTPUT.mkdir(exist_ok=False)
    source = ROOT / 'BalloonPoppingGymEnv/agents/scenario4_proactive_joint_agent.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = dict(seeds=SEEDS, stage='development, not held-out',
                reference_files=[str(REFERENCE / f'{s}_joint.json') for s in SEEDS],
                source_sha256=digest, cpu_workers=2, gpu_visible=False,
                gate='total gain >=2, no seed loss >1, accepted proactive repair >0',
                auto_promote=False)
    (OUTPUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'), str(ROOT))))

    def flight(seed):
        config = dict(references[seed]['config'])
        config.update(agent_module_path=str(source.relative_to(ROOT)),
                      agent_class_name='Scenario4ProactiveJointAgent',
                      agent_name='Development proactive joint refresh',
                      record_guidance_telemetry=True)
        path = OUTPUT / f'{seed}_proactive.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        report = OUTPUT / f'{seed}_proactive.json'
        with (OUTPUT / f'{seed}_proactive.log').open('x', encoding='utf-8') as log:
            subprocess.run((sys.executable, '-m', 'scripts.evaluate_guidance', str(path),
                            '--seeds', str(seed), '--report', str(report)),
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=2400)
        row = read_flight(report)
        if row['seed'] != seed or row['agent_sha256'] != digest:
            raise ValueError('Source or seed mismatch')
        return seed, dict(reference=summarize(references[seed]), candidate=summarize(row),
                          proactive={k: v for k, v in row['diagnostics'].items()
                                     if k.startswith('proactive_')})

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(flight, s) for s in SEEDS]):
                seed, result = future.result()
                results[str(seed)] = result
                (OUTPUT / 'progress.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(f'COMPLETE seed={seed}: {result}', flush=True)
        deltas = [results[str(s)]['candidate']['score'] - results[str(s)]['reference']['score']
                  for s in SEEDS]
        accepted = sum(r['proactive']['proactive_accepted'] for r in results.values())
        summary = dict(results=results, deltas=deltas, total_gain=sum(deltas),
                       passed=sum(deltas) >= 2 and min(deltas) >= -1 and accepted > 0,
                       stage='development only; independent validation required if passed',
                       live_agent_changed=False)
        (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        (OUTPUT / 'failed.json').write_text(json.dumps(dict(error=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
