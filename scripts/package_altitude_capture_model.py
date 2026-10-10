"""Freeze a seed-free seven-feature model from the first audited pool only.

This creates an *experimental* candidate, not an approved live model. The
feature was selected on this development pool, so it still needs the disjoint
second-pool calibration screen and then a separate flight-score holdout.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_altitude_route_agent import (
    ALTITUDE_FEATURES, FrozenAltitudeCaptureModel,
)
from scripts.diagnose_detailed_route_capture import (
    DETAIL_NAMES, load_detailed_audited_attempts,
)
from scripts.diagnose_route_capture_ml import fit_regularized_logistic
from scripts.package_guarded_capture_model import TRAINING_SEEDS


def package(archive_path, output_path):
    archive_path, output_path = Path(archive_path), Path(output_path)
    if output_path.exists() or archive_path.resolve() == output_path.resolve():
        raise ValueError('A distinct new model path is required')
    records = load_detailed_audited_attempts(archive_path)
    if {row['seed'] for row in records} != set(TRAINING_SEEDS):
        raise ValueError('Unexpected first-pool audited field set')
    selected = list(range(6)) + [DETAIL_NAMES.index('relative_altitude')]
    x = np.asarray([row['features'][selected] for row in records])
    y = np.asarray([row['hit'] for row in records])
    model = fit_regularized_logistic(x, y)
    payload = dict(
        schema='altitude_capture_risk_experimental_v1',
        feature_names=list(ALTITUDE_FEATURES),
        center=model[0].tolist(), scale=model[1].tolist(),
        coefficients=model[2].tolist(),
        source_archive_sha256=hashlib.sha256(
            archive_path.read_bytes()).hexdigest(),
        development_fields=len(TRAINING_SEEDS),
        development_attempts=len(records),
        role='experimental only; never auto-promoted',
    )
    FrozenAltitudeCaptureModel(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + '\n',
                           encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    model = package(args.archive, args.output)
    print(json.dumps(dict(fields=model['development_fields'],
                          attempts=model['development_attempts'],
                          source_sha256=model['source_archive_sha256'])))


if __name__ == '__main__':
    main()
