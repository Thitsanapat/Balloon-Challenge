"""Paired development screen for observation-only timing rescue; no upload."""
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
OUTPUT = RESULTS / 'timed_rescue_dev_20261010'
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
    source = ROOT / 'BalloonPoppingGymEnv/agents/scenario4_timed_rescue_agent.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = dict(seeds=SEEDS, scenario=4, stage='development, not held-out',
                source_sha256=source_hash, reference_scores=[6, 4],
                cpu_workers=2, gpu_visible=False, auto_promote=False,
                gate='total gain >=2, no seed loss >1, accepted delayed rescue >0')
    (OUTPUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'), str(ROOT))))

    def flight(seed):
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('Candidate source changed during comparison')
        config = dict(references[seed]['config'])
        config.update(agent_module_path=str(source.relative_to(ROOT)),
                      agent_class_name='Scenario4TimedRescueAgent',
                      agent_name='Development: delayed joint rescue',
                      record_guidance_telemetry=True)
        path = OUTPUT / f'{seed}_candidate.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        report = OUTPUT / f'{seed}_candidate.json'
        with (OUTPUT / f'{seed}_candidate.log').open('x', encoding='utf-8') as log:
            subprocess.run((sys.executable, '-m', 'scripts.evaluate_guidance', str(path),
                            '--seeds', str(seed), '--report', str(report)),
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=2400, check=True)
        row = read_flight(report)
        if row['seed'] != seed or row['agent_sha256'] != source_hash:
            raise ValueError('Unexpected seed or source hash')
        return seed, dict(reference=summarize(references[seed]), candidate=summarize(row),
                          timed={key: value for key, value in row['diagnostics'].items()
                                 if key.startswith('timed_rescue_')})

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(flight, seed) for seed in SEEDS]):
                seed, result = future.result()
                results[str(seed)] = result
                (OUTPUT / 'progress.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(f'COMPLETE seed={seed} reference={result["reference"]["score"]} '
                      f'candidate={result["candidate"]["score"]} '
                      f'timed={result["timed"]}', flush=True)
        differences = [results[str(s)]['candidate']['score'] -
                       results[str(s)]['reference']['score'] for s in SEEDS]
        accepted = sum(r['timed']['timed_rescue_accepted'] for r in results.values())
        summary = dict(results=results, differences=differences,
                       total_gain=sum(differences),
                       passed=sum(differences) >= 2 and min(differences) >= -1 and
                       accepted > 0,
                       stage='development only; fresh paired validation if passed',
                       live_agent_changed=False)
        (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(dict(differences=differences, passed=summary['passed'])), flush=True)
    except Exception as error:
        (OUTPUT / 'failed.json').write_text(json.dumps(dict(error=str(error))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
