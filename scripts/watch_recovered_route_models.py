"""Fit predeclared offline route models after the audited recovery finishes."""

import json
import time

from scripts.fit_combined_guarded_route import fit
from scripts.qualification_run import ROOT
from scripts.validate_detailed_capture_holdout import validate


RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
FIRST = RESULTS / 'guarded_route_train12_20261008.zip'
RECOVERED_DIR = RESULTS / 'guarded_route_train12_b_recovered_20261009'
SECOND = RESULTS / 'guarded_route_train12_b_recovered_20261009.zip'


def run(max_seconds=8 * 3600):
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if (RECOVERED_DIR / 'failed.json').exists():
            raise RuntimeError('Recovered audited collection failed')
        if SECOND.exists() and (RECOVERED_DIR / 'route_capture_fit.json').exists():
            summary = json.loads((RECOVERED_DIR / 'summary.json').read_text())
            if not summary.get('all_passed') or len(summary.get('completed', ())) != 12:
                raise RuntimeError('Recovered collection was not fully audited')
            fit_report = fit(
                FIRST, SECOND,
                RESULTS / 'guarded_route_combined24_recovered_20261009_fit.json',
                RESULTS / 'guarded_route_combined24_recovered_20261009_weights.json')
            detailed = validate(FIRST, SECOND)
            detailed_path = RESULTS / 'guarded_route_altitude_holdout12_recovered_20261009.json'
            if detailed_path.exists():
                raise FileExistsError(detailed_path)
            detailed_path.write_text(
                json.dumps(detailed, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(dict(
                pooled_fields=fit_report['fields'],
                pooled_attempts=fit_report['attempts'],
                pooled_six_feature_gate=fit_report['predeclared_gate']['passed'],
                disjoint_altitude_gate=detailed['calibration_gate_passed'],
                disjoint_brier_gain=detailed['brier_gain'],
                disjoint_log_loss_gain=detailed['log_loss_gain'])), flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Recovered route pool did not finish within eight hours')


if __name__ == '__main__':
    run()
