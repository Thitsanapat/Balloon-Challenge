"""The paired summary must not treat incomplete or mismatched runs as gains."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_paired_evaluations import summarize


def _result(seed, score, source, *, truncated=False, config=None):
    return {
        "scenario": 4, "seed": seed, "score": score,
        "wall_seconds": 300.0 + seed,
        "terminated": not truncated, "truncated": truncated,
        "agent_sha256": source,
        "agent_file_unchanged_during_run": True,
        "agent_source_dependencies_sha256": {"agent.py": source},
        "agent_sources_unchanged_during_run": True,
        "config": config or {"agent_kwargs": {"launch_time": 24.0}},
    }


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def _write(self, name, rows):
        path = self.directory / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_pairs_multiple_reports_by_seed_and_summarizes_only_actual_scores(self):
        old = [self._write("old1.json", [_result(11, 1, "old")]),
               self._write("old2.json", [_result(10, 3, "old")])]
        new = [self._write("new1.json", [_result(10, 5, "new"),
                                           _result(11, 0, "new")])]
        summary = summarize(old, new)
        self.assertEqual([row["seed"] for row in summary["pairs"]], [10, 11])
        self.assertEqual([row["score_delta"] for row in summary["pairs"]], [2, -1])
        self.assertEqual(summary["mean_score_delta"], 0.5)
        self.assertEqual(summary["mean_baseline_score"], 2)
        self.assertEqual(summary["mean_candidate_score"], 2.5)
        self.assertEqual(summary["mean_baseline_wall_seconds"], 310.5)
        self.assertEqual(summary["mean_candidate_wall_seconds"], 310.5)
        self.assertEqual((summary["wins"], summary["ties"], summary["losses"]), (1, 0, 1))
        self.assertTrue(summary["all_pairs_complete"])
        self.assertEqual(summary["baseline_source_sha256"], "old")
        self.assertEqual(summary["candidate_source_sha256"], "new")

    def test_truncation_cannot_create_a_comparison_gain(self):
        old = self._write("old.json", [_result(10, 1, "old")])
        new = self._write("new.json", [_result(10, 9, "new", truncated=True)])
        summary = summarize([old], [new])
        self.assertFalse(summary["all_pairs_complete"])
        self.assertEqual(summary["complete_pair_count"], 0)
        self.assertIsNone(summary["mean_score_delta"])
        self.assertIsNone(summary["total_score_delta"])
        self.assertIsNone(summary["pairs"][0]["score_delta"])
        self.assertTrue(summary["pairs"][0]["candidate_truncated"])

    def test_mismatched_or_duplicate_seeds_are_rejected(self):
        old = self._write("old.json", [_result(10, 1, "old")])
        new = self._write("new.json", [_result(11, 2, "new")])
        with self.assertRaisesRegex(ValueError, "seed sets differ"):
            summarize([old], [new])
        duplicate = self._write("duplicate.json", [_result(10, 2, "old")])
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            summarize([old, duplicate], [new])

    def test_changed_config_is_rejected(self):
        old = self._write("old.json", [_result(10, 1, "old"),
                                       _result(11, 2, "old", config={"launch_time": 42})])
        new = self._write("new.json", [_result(10, 2, "new"),
                                       _result(11, 3, "new")])
        with self.assertRaisesRegex(ValueError, "config, or source changed"):
            summarize([old], [new])

    def test_changed_source_and_in_run_source_mutation_are_not_promoted(self):
        old = self._write("old.json", [_result(10, 1, "old"),
                                       _result(11, 2, "changed")])
        new = self._write("new.json", [_result(10, 2, "new"),
                                       _result(11, 3, "new")])
        with self.assertRaisesRegex(ValueError, "config, or source changed"):
            summarize([old], [new])

        old = self._write("old.json", [_result(10, 1, "old")])
        mutated = _result(10, 9, "new")
        mutated["agent_sources_unchanged_during_run"] = False
        new = self._write("new.json", [mutated])
        summary = summarize([old], [new])
        self.assertFalse(summary["all_pairs_complete"])
        self.assertIsNone(summary["mean_score_delta"])
        self.assertFalse(summary["pairs"][0]["candidate_sources_unchanged"])


if __name__ == "__main__":
    unittest.main()
