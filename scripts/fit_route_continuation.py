"""Small CPU calibration pilot for *factual* route continuation.

Train on one archived set of fields and test on a disjoint set.  This does
not estimate outcomes for unflown route alternatives.  No weights are
exported to a flight agent, because these archives used the older guarded
policy and their second-leg outcomes are selectively observed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.diagnose_route_capture_ml import (
    fit_regularized_logistic, predict_logistic,
)


FEATURE_NAMES = ("planned_route_length", "planned_first_seconds",
                 "planned_second_seconds", "planned_departure_cosine")


def features(rows):
    return np.asarray([[float(row[name] if row[name] is not None else 0.)
                        for name in FEATURE_NAMES] for row in rows], dtype=float)


def metrics(probability, observed):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(observed, dtype=float)
    return dict(brier=float(np.mean((p-y)**2)),
                log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))))


def transfer(train, test, label):
    train = [row for row in train if row[label] is not None]
    test = [row for row in test if row[label] is not None]
    if len(train) < 20 or len(test) < 20:
        raise ValueError("Too few factual labels for a transfer check")
    if {row["seed"] for row in train} & {row["seed"] for row in test}:
        raise ValueError("Training and test fields must be disjoint")
    y_train = np.asarray([row[label] for row in train], dtype=float)
    y_test = np.asarray([row[label] for row in test], dtype=float)
    model = fit_regularized_logistic(features(train), y_train,
                                     penalty=8., iterations=30)
    prediction = predict_logistic(features(test), model)
    prior = np.full(len(test), (y_train.sum()+.5)/(len(train)+1.))
    return dict(train_fields=len({row["seed"] for row in train}),
                test_fields=len({row["seed"] for row in test}),
                train_labels=len(train), test_labels=len(test),
                test_hits=int(y_test.sum()), prior=metrics(prior, y_test),
                learned=metrics(prediction, y_test))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_a", type=Path)
    parser.add_argument("archive_b", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite existing report")
    a = json.loads(args.archive_a.read_text(encoding="utf-8"))["rows"]
    b = json.loads(args.archive_b.read_text(encoding="utf-8"))["rows"]
    result = dict(scope="Old guarded policy only; not a submission model",
                  feature_names=FEATURE_NAMES,
                  labels={label: {"a_to_b": transfer(a, b, label),
                                  "b_to_a": transfer(b, a, label)}
                          for label in ("first_hit", "second_hit")})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["labels"]))


if __name__ == "__main__":
    main()
