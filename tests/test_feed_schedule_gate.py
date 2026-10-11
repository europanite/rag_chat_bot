"""Regression tests for the scheduled feed gate. Run: python -m unittest discover -s tests."""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'feed_schedule_gate.py'
spec = importlib.util.spec_from_file_location('feed_schedule_gate', MODULE_PATH)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

WINDOW_ARGS = {
    'api_url': 'https://api.github.com',
    'repo': 'europanite/rag_chat_bot',
    'workflow_file': 'feed.yml',
    'token': 'example-token',
    'current_run_id': 999,
    'start_utc': datetime(2026, 10, 10, 21, tzinfo=UTC),
    'end_utc': datetime(2026, 10, 11, 0, tzinfo=UTC),
}


def make_run(run_id=123, status='completed', conclusion='success'):
    return {'id': run_id, 'created_at': '2026-10-10T21:17:00Z',
            'status': status, 'conclusion': conclusion}


def pages_job(published=True):
    return {
        'name': 'sweather / pages / build',
        'conclusion': 'success',
        'steps': [
            {'name': 'Restore generated state artifact into public dir (optional)',
             'conclusion': 'success' if published else 'skipped'},
            {'name': 'Deploy frontend shell to S3 (preserve generated feed/image history)',
             'conclusion': 'success'},
        ],
    }


class FeedGateTests(unittest.TestCase):
    def find(self, *, runs, jobs=None):
        def fake_request(url, token):
            if '/jobs?' in url:
                result = jobs if jobs is not None else []
                return {'total_count': len(result), 'jobs': result}
            return {'workflow_runs': runs}

        with patch.object(gate, 'request_json', side_effect=fake_request):
            return gate.find_blocking_run(**WINDOW_ARGS)

    def test_green_select_only_does_not_block(self):
        self.assertEqual(self.find(runs=[make_run()], jobs=[
            {'name': 'select', 'conclusion': 'success', 'steps': []}
        ]), (False, 'no_prior_publication'))

    def test_actual_published_pages_run_blocks(self):
        self.assertEqual(self.find(runs=[make_run()], jobs=[pages_job()]),
                         (True, 'already_published'))

    def test_pages_without_generated_state_does_not_block(self):
        self.assertEqual(self.find(runs=[make_run()], jobs=[pages_job(False)]),
                         (False, 'no_prior_publication'))

    def test_failed_after_s3_deploy_also_blocks(self):
        job = pages_job()
        job['conclusion'] = 'failure'  # e.g. CloudFront invalidation failed
        self.assertEqual(self.find(runs=[make_run(conclusion='failure')], jobs=[job]),
                         (True, 'already_published'))

    def test_api_error_is_not_a_successful_skip(self):
        with patch.object(gate, 'request_json', side_effect=OSError('API unreachable')):
            with self.assertRaisesRegex(OSError, 'API unreachable'):
                gate.find_blocking_run(**WINDOW_ARGS)

    def test_queued_or_running_run_blocks(self):
        self.assertEqual(self.find(runs=[make_run(status='in_progress', conclusion=None)]),
                         (True, 'already_in_progress'))

    def test_failed_run_does_not_block(self):
        self.assertEqual(self.find(runs=[make_run(conclusion='failure')]),
                         (False, 'no_prior_publication'))

    def test_previous_day_not_block(self):
        run = make_run()
        run['created_at'] = '2026-10-10T15:17:00Z'
        self.assertEqual(self.find(runs=[run], jobs=[pages_job()]),
                         (False, 'no_prior_publication'))

    def test_missing_token_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, 'GITHUB_TOKEN'):
            gate.find_blocking_run(**{**WINDOW_ARGS, 'token': ''})

    def test_incomplete_jobs_fail_closed(self):
        with patch.object(gate, 'request_json', return_value={
            'total_count': 101, 'jobs': [pages_job()]
        }):
            with self.assertRaisesRegex(RuntimeError, 'incomplete job list'):
                gate.was_feed_published(api_url='https://api.github.com',
                                        repo='europanite/rag_chat_bot', run_id=123,
                                        token='example-token')

    def test_emit_logs_and_writes_outputs(self):
        import tempfile
        d = gate.Decision(False, 'weather', '2026-10-11-morning', 'already_published')
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = str(Path(tmpdir) / 'github_output')
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                gate.emit(d, dest)
            self.assertIn('reason=already_published', stdout.getvalue())
            self.assertEqual(stdout.getvalue(), Path(dest).read_text())


if __name__ == '__main__':
    unittest.main()
