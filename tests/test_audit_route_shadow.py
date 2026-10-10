import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_route_shadow import audit, audit_rows
from tests.test_route_shadow import controller, observation
from BalloonPoppingGymEnv.training.route_shadow import RouteShadowCollector


class RouteAuditTests(unittest.TestCase):
    def rows(self):
        rows = []
        collector = RouteShadowCollector(rows.append)
        agent = controller()
        collector.after_action(agent, observation())
        agent.plan_start = 1.5
        agent.plan_duration = 1.5
        collector.after_action(agent, observation(1.5))
        collector.finish(observation(2., (2, 1, 0)))
        return rows

    def test_revisions_are_one_attempt_not_independent_labels(self):
        report = audit_rows(self.rows())
        self.assertEqual(report['decisions'], 2)
        self.assertEqual(report['executed_attempts'], 1)
        self.assertEqual(report['imitation_decision_ids'], [0])
        self.assertEqual(report['deadline_outcome_attempt_ids'], [0])

    def test_fake_labels_hidden_data_or_duplicate_outcomes_rejected(self):
        original = self.rows()
        for damage in ('hidden', 'fake_label', 'duplicate', 'time', 'deadline'):
            rows = copy.deepcopy(original)
            if damage == 'hidden':
                rows[1]['visible_states'][2][0] = 1.
            elif damage == 'fake_label':
                rows[1]['route_prefixes'][0]['actual_hits'] = 0
            elif damage == 'duplicate':
                rows.append(rows[-1])
            elif damage == 'deadline':
                rows[-2]['outcome'] = 'miss'
            else:
                rows[-1]['time'] = 0.
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                audit_rows(rows)

    def test_censoring_not_used_as_failed_hit_label(self):
        rows = []
        collector = RouteShadowCollector(rows.append)
        collector.after_action(controller(), observation())
        collector.finish(observation(2.))
        report = audit_rows(rows)
        self.assertEqual(report['deadline_outcome_attempt_ids'], [])
        with self.assertRaises(ValueError):
            audit_rows(rows[:-1])

    def test_file_audit_requires_hash_parity_and_matching_counters(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            data = directory/'decisions.jsonl'
            data.write_text(''.join(json.dumps(r) + '\n' for r in self.rows()), encoding='utf-8')
            result = dict(action_sha256='a', observation_sha256='b', steps=50,
                          score=1, final_time=5., terminated=True, truncated=False,
                          collection_counts=self.rows()[-1]['counts'])
            gate = dict(exact_parity=True, baseline=result, shadow=copy.deepcopy(result),
                        dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
            gate_path = directory/'gate.json'
            gate_path.write_text(json.dumps(gate), encoding='utf-8')
            self.assertTrue(audit(directory)['parity_verified'])
            for field in ('hash', 'actions', 'counts'):
                broken = copy.deepcopy(gate)
                if field == 'hash':
                    broken['dataset_sha256'] = 'wrong'
                elif field == 'actions':
                    broken['shadow']['action_sha256'] = 'changed'
                else:
                    broken['shadow']['collection_counts']['miss'] = 1
                gate_path.write_text(json.dumps(broken), encoding='utf-8')
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit(directory)

    def test_all_pops_complete_and_unique(self):
        rows = []
        collector = RouteShadowCollector(rows.append)
        collector.after_action(controller(), observation())
        collector.finish(observation(2., (2, 2, 0)))
        report = audit_rows(rows)
        self.assertEqual(report['pop_counts']['observed_pops'], 2)
        self.assertEqual(report['pop_counts']['incidental_pops'], 1)
        self.assertTrue(report['all_pop_events_complete'])
        for damage in ('drop_pop', 'duplicate_pop', 'wrong_relation', 'wrong_final', 'no_end', 'no_start'):
            broken = copy.deepcopy(rows)
            pop_index = next(i for i, r in enumerate(broken) if r['kind'] == 'pop')
            if damage == 'drop_pop':
                del broken[pop_index]
            elif damage == 'duplicate_pop':
                broken.insert(pop_index, copy.deepcopy(broken[pop_index]))
            elif damage == 'wrong_relation':
                broken[pop_index]['relation'] = 'unassigned'
            elif damage == 'wrong_final':
                broken[-1]['final_popped_ids'] = [0]
            elif damage == 'no_end':
                broken.pop()
            else:
                broken.pop(0)
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                audit_rows(broken)

    def test_empty_zero_pop_episode_is_valid_not_dropped(self):
        rows = []
        collector = RouteShadowCollector(rows.append)
        collector.observe(observation(0.))
        collector.finish(observation(1.))
        report = audit_rows(rows)
        self.assertTrue(report['all_pop_events_complete'])
        self.assertEqual(report['executed_attempts'], 0)

    def test_legacy_rows_remain_readable_but_not_all_pop_complete(self):
        legacy = [r for r in self.rows() if r['kind'] in ('decision', 'outcome')]
        report = audit_rows(legacy)
        self.assertFalse(report['all_pop_events_complete'])
        self.assertEqual(report['outcomes'], {'hit': 1})


if __name__ == '__main__':
    unittest.main()
