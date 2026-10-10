"""Recover an audited route-shadow pool after a source-hash stop.

Never changes or deletes the failed original.  Copies only fully audited
completed pairs into a new output, then runs every remaining seed twice in
fresh official processes under the *original* frozen collector manifest.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from scripts.audit_route_shadow import audit
from scripts.collect_guarded_route_shadow import ROOT, source_manifest
from scripts.collect_route_shadow import paired_gate
from scripts.diagnose_route_capture_ml import (
    fit_regularized_logistic, held_out_report, load_audited_attempts,
)
from scripts.fit_combined_guarded_route import SECOND_SEEDS


def recover(original, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    if (not original.is_relative_to(ROOT) or not output.is_relative_to(ROOT)
            or original == output or output.exists() or
            output.with_suffix('.zip').exists()):
        raise ValueError('Distinct new in-workspace recovery output required')
    plan = json.loads((original / 'plan.json').read_text(encoding='utf-8'))
    progress = json.loads((original / 'progress.json').read_text(encoding='utf-8'))
    seeds = tuple(plan['seeds'])
    completed = progress['completed']
    if (seeds != SECOND_SEEDS or
            tuple(row['seed'] for row in completed) != seeds[:len(completed)] or
            not (original / 'failed.json').exists()):
        raise ValueError('Not the expected audited partial second pool')
    config_path = ROOT / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml'
    _, manifest = source_manifest(config_path)
    if manifest != plan['source_manifest']:
        raise ValueError('Frozen collector source does not match original manifest')
    output.mkdir()
    recovered_plan = dict(plan, recovery_of=str(original),
                          original_failed_marker_retained=True,
                          copied_verified_pairs=len(completed))
    (output / 'plan.json').write_text(
        json.dumps(recovered_plan, indent=2), encoding='utf-8')
    for row in completed:
        seed = row['seed']
        source = original / str(seed)
        copied = output / str(seed)
        pair = {label: json.loads((source / f'{label}.json').read_text())
                for label in ('baseline', 'shadow')}
        if (not paired_gate(pair['baseline'], pair['shadow']) or
                any(flight['source_manifest'] != manifest for flight in pair.values()) or
                json.loads(json.dumps(audit(source))) !=
                json.loads((source / 'audit.json').read_text()) or
                hashlib.sha256((source / 'audit.json').read_bytes()).hexdigest()
                != row['audit_sha256']):
            raise ValueError(f'Copied field {seed} did not re-audit')
        shutil.copytree(source, copied)
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                     MKL_NUM_THREADS='1',
                     PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'),
                                                 str(ROOT))))
    try:
        for seed in seeds[len(completed):]:
            if source_manifest(config_path)[1] != manifest:
                raise RuntimeError('Frozen source changed before seed')
            folder = output / str(seed)
            folder.mkdir()
            for label in ('baseline', 'shadow'):
                command = (sys.executable, '-m',
                           'scripts.collect_guarded_route_shadow',
                           '--config', str(config_path), '--output', str(folder),
                           '--seeds', str(seed), '--flight', label)
                with (folder / f'{label}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=child_env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1500)
            pair = {label: json.loads((folder / f'{label}.json').read_text())
                    for label in ('baseline', 'shadow')}
            if not paired_gate(pair['baseline'], pair['shadow']):
                raise RuntimeError(f'Action/observation parity failed on {seed}')
            gate = dict(seed=seed, exact_parity=True,
                        dataset_sha256=hashlib.sha256(
                            (folder / 'decisions.jsonl').read_bytes()).hexdigest(),
                        **pair)
            (folder / 'gate.json').write_text(
                json.dumps(gate, indent=2), encoding='utf-8')
            audited = audit(folder)
            if not audited['all_pop_events_complete']:
                raise RuntimeError(f'Incomplete pop audit on {seed}')
            (folder / 'audit.json').write_text(
                json.dumps(audited, indent=2), encoding='utf-8')
            completed.append(dict(
                seed=seed, score=pair['shadow']['score'],
                attempts=audited['executed_attempts'],
                audit_sha256=hashlib.sha256(
                    (folder / 'audit.json').read_bytes()).hexdigest()))
            (output / 'progress.json').write_text(json.dumps(dict(
                completed=completed, completed_pairs=len(completed),
                planned_pairs=len(seeds)), indent=2), encoding='utf-8')
            print(f'RECOVERED seed={seed} score={pair["shadow"]["score"]} '
                  f'completed={len(completed)}/{len(seeds)}', flush=True)
        (output / 'summary.json').write_text(json.dumps(dict(
            completed=completed, all_passed=True,
            recovery_of=str(original)), indent=2), encoding='utf-8')
        archive = output.with_suffix('.zip')
        with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as zip_file:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    zip_file.write(path, 'data/' + str(
                        path.relative_to(output)).replace('\\', '/'))
        records = load_audited_attempts(archive)
        report = held_out_report(records)
        model = fit_regularized_logistic(
            np.asarray([row['features'] for row in records]),
            np.asarray([row['hit'] for row in records]))
        report['full_fit_prototype'] = dict(
            center=model[0].tolist(), scale=model[1].tolist(),
            coefficients=model[2].tolist(), role='offline prototype only')
        report['source_archive_sha256'] = hashlib.sha256(
            archive.read_bytes()).hexdigest()
        (output / 'route_capture_fit.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        print(f'COMPLETE_RECOVERY fields={len(seeds)} attempts={len(records)}',
              flush=True)
    except Exception as exc:
        (output / 'failed.json').write_text(json.dumps(dict(
            error=str(exc), completed_pairs=len(completed)), indent=2),
            encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    recover(args.original, args.output)


if __name__ == '__main__':
    main()
