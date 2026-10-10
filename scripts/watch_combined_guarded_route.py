"""Wait for the audited second pool, then fit the frozen 24-field model once."""

import json
import time

from scripts.fit_combined_guarded_route import fit
from scripts.qualification_run import ROOT


RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
FIRST = RESULTS / 'guarded_route_train12_20261008.zip'
SECOND_DIR = RESULTS / 'guarded_route_train12_b_20261009'
SECOND = RESULTS / 'guarded_route_train12_b_20261009.zip'
REPORT = RESULTS / 'guarded_route_combined24_20261009_fit.json'
WEIGHTS = RESULTS / 'guarded_route_combined24_20261009_weights.json'


def run(max_seconds=8 * 3600):
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if (SECOND_DIR / 'failed.json').exists():
            raise RuntimeError('Second route-data collection failed')
        if (SECOND_DIR / 'route_capture_fit.json').exists() and SECOND.exists():
            summary = json.loads((SECOND_DIR / 'summary.json').read_text())
            if not summary.get('all_passed') or len(summary.get('completed', ())) != 12:
                raise RuntimeError('Incomplete audited training pool')
            report = fit(FIRST, SECOND, REPORT, WEIGHTS)
            print(json.dumps(dict(fields=report['fields'],
                                  attempts=report['attempts'],
                                  prior=report['prior'], learned=report['learned'],
                                  calibration_passed=report['predeclared_gate']['passed'])),
                  flush=True)
            return report
        time.sleep(30)
    raise TimeoutError('Second route-data pool did not finish within eight hours')


if __name__ == '__main__':
    run()
