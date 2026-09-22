"""Submission preparation must not package regressions or leak credentials."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts import run_submission


def runner_fixture(tmp_path, monkeypatch, score=7):
    source = tmp_path / 'agent.py'
    source.write_text('# test agent\n', encoding='utf-8')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(
        agent_module_path=str(source), agent_class_name='Agent',
        agent_name='test', agent_kwargs={}, scenario_number=1, render_mode=None,
    )), encoding='utf-8')
    env = SimpleNamespace(_popped_count=score, np_random_seed=0, close=Mock())
    monkeypatch.setattr(run_submission, '_load_agent_class', lambda *a: object)
    monkeypatch.setattr(run_submission, 'evaluate_scenario', lambda *a, **k: (env, None, {}))
    packer = Mock()
    monkeypatch.setattr(run_submission, 'pack_for_submission', packer)
    return config, env, packer


def test_dry_run_evaluates_without_packaging(tmp_path, monkeypatch):
    config, env, packer = runner_fixture(tmp_path, monkeypatch)
    report = tmp_path / 'receipt.json'
    monkeypatch.setattr('sys.argv', ['run_submission', str(config), '--dry-run',
                                    '--report', str(report), '--min-score', '7'])
    run_submission.main()
    assert json.loads(report.read_text())['score'] == 7
    packer.assert_not_called()
    env.close.assert_called_once()


def test_low_score_cannot_be_packaged(tmp_path, monkeypatch):
    config, env, packer = runner_fixture(tmp_path, monkeypatch, score=6)
    monkeypatch.setenv('BPC_TEAM_NAME', 'test-team')
    monkeypatch.setenv('BPC_TEAM_SECRET', 'test-secret')
    monkeypatch.setattr('sys.argv', ['run_submission', str(config), '--min-score', '7'])
    with pytest.raises(RuntimeError, match='not packaged'):
        run_submission.main()
    packer.assert_not_called()
    env.close.assert_called_once()


def test_receipt_omits_credentials_from_existing_submission(tmp_path, monkeypatch):
    config, _, packer = runner_fixture(tmp_path, monkeypatch)
    previous = tmp_path / 'previous.json'
    previous.write_text(json.dumps({'team': {'name': 'test-team', 'secret': 'test-secret'}}))
    report = tmp_path / 'receipt.json'
    monkeypatch.setattr('sys.argv', ['run_submission', str(config), str(previous),
                                    '--report', str(report)])
    run_submission.main()
    assert 'test-secret' not in report.read_text()
    assert 'test-team' not in report.read_text()
    assert packer.call_args.args[0]['team_secret'] == 'test-secret'
