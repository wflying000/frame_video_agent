import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import consistency, model, prompts, video_agent
from agent.common import digest, file_hash, read_json, write_json
from agent.store import Blocked, Project


class ReviewConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        self.config = {'story': 'A fish finds its way home.'}
        settings = patch.object(model, 'settings', return_value={})
        settings.start()
        self.addCleanup(settings.stop)

    @staticmethod
    def action(directory, version, previous):
        (directory / 'scene.html').write_text('same original scene', encoding='utf-8')
        return {'version': version}

    def make_legacy(self):
        manifest = read_json(self.root / 'manifest.json')
        for key in ('review_policy', 'render_consistency', 'legacy_policy_compat',
                    'legacy_review_compat', 'legacy_render_compat'):
            manifest.pop(key, None)
        for node in manifest['tasks'].values():
            signature = digest({'inputs': {}, 'dependencies': node['input_versions'], 'model': {},
                'prompts': {key: value for key, value in vars(prompts).items() if key.isupper()}})
            node['input_hash'] = signature
            for attempt in node['attempts']:
                attempt['input_hash'] = signature
                attempt.pop('review_policy', None)
                attempt.pop('render_consistency', None)
        write_json(self.root / 'manifest.json', manifest)

    def test_new_defaults_and_content_config_are_independent(self):
        with Project(self.root, self.config) as project:
            self.assertEqual(project.review_strictness, 'balanced')
            self.assertEqual(project.render_consistency, 'exact')
            self.assertEqual(project.data['config_hash'], digest(self.config))
            project.task('segments/S01', {}, [], self.action)
            candidate = project.data['tasks']['segments/S01']['attempts'][0]
            self.assertEqual(candidate['review_policy'], consistency.review_policy('balanced'))
            self.assertEqual(candidate['render_consistency'], 'exact')
        self.assertEqual(read_json(self.root / 'project_config.json'), self.config)

    def test_resume_inherits_saved_policy_and_explicit_override_wins(self):
        with Project(self.root, self.config, review_strictness='relaxed', render_consistency='tolerant'):
            pass
        with Project(self.root, self.config, resume=True) as project:
            self.assertEqual((project.review_strictness, project.render_consistency), ('relaxed', 'tolerant'))
            self.assertNotIn('policy_history', project.data)
        with Project(self.root, self.config, resume=True, review_strictness='balanced') as project:
            self.assertEqual((project.review_strictness, project.render_consistency), ('balanced', 'tolerant'))
            self.assertEqual(project.data['policy_history'][0]['from']['review_policy']['name'], 'relaxed')

    def test_unspecified_legacy_resume_keeps_strict_exact_and_pass_cache(self):
        with Project(self.root, self.config, review_strictness='strict') as project:
            project.task('segments/S01', {}, [], self.action)
        self.make_legacy()
        for _ in range(2):
            with Project(self.root, self.config, resume=True) as project:
                self.assertEqual((project.review_strictness, project.render_consistency), ('strict', 'exact'))
                self.assertTrue(project.data['legacy_policy_compat'])
                project.task('segments/S01', {}, [], lambda *args: self.fail('legacy PASS reran'))

    def test_review_migration_preserves_legacy_assets_and_invalidates_only_review(self):
        with Project(self.root, self.config, review_strictness='strict') as project:
            project.task('assets/FISH', {}, [], self.action)
            project.task('segments/S01', {}, ['assets/FISH'], self.action)
        self.make_legacy()
        with Project(self.root, self.config, resume=True, review_strictness='balanced') as project:
            self.assertEqual(project.data['tasks']['assets/FISH']['status'], 'PASS')
            self.assertEqual(project.data['tasks']['segments/S01']['status'], 'STALE')
            project.task('assets/FISH', {}, [], lambda *args: self.fail('asset reran after review change'))
            project.task('segments/S01', {}, ['assets/FISH'], self.action)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 2)

    def test_policy_change_keeps_budgets_history_and_archives_final(self):
        with Project(self.root, self.config, review_strictness='strict') as project:
            project.task('plans/story', {}, [], self.action)
            project.task('segments/S01', {}, ['plans/story'], self.action)
            project.task('checks/timeline', {}, ['segments/S01'], self.action)
            project.task('exports/final', {}, ['checks/timeline'], self.action)
            project.data.update(calls=37, active_seconds=100, global_repairs=1)
        old_scene = file_hash(self.root / 'segments/S01/v001/scene.html')
        (self.root / 'final.mp4').write_bytes(b'old verified export')
        with Project(self.root, self.config, resume=True, review_strictness='balanced') as project:
            self.assertEqual(project.data['calls'], 37)
            self.assertGreaterEqual(project.data['active_seconds'], 100)
            self.assertEqual(project.data['global_repairs'], 1)
            self.assertEqual(project.data['tasks']['plans/story']['status'], 'PASS')
            history = project.data['policy_history'][0]
            self.assertEqual(set(history['affected']), {'segments/S01', 'checks/timeline', 'exports/final'})
            self.assertIn('time', history)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)
            self.assertFalse((self.root / 'final.mp4').exists())
            self.assertEqual(len(list((self.root / 'exports').glob('previous_final_*.mp4'))), 1)
        self.assertEqual(file_hash(self.root / 'segments/S01/v001/scene.html'), old_scene)

    def test_render_change_invalidates_assets_and_dependents(self):
        with Project(self.root, self.config) as project:
            project.task('assets/FISH', {}, [], self.action)
            project.task('checks/animatic', {}, ['assets/FISH'], self.action)
            project.task('segments/S01', {}, ['assets/FISH'], self.action)
        with Project(self.root, self.config, resume=True, render_consistency='tolerant') as project:
            self.assertEqual(set(project.data['policy_history'][0]['affected']),
                             {'assets/FISH', 'checks/animatic', 'segments/S01'})
            self.assertTrue(all(node['status'] == 'STALE' for node in project.data['tasks'].values()))

    def test_old_repairs_are_archived_and_only_pending_local_changes_remain(self):
        with Project(self.root, self.config, review_strictness='strict') as project:
            for shot in ('S01', 'S02', 'S03', 'S04'):
                project.task(f'segments/{shot}', {}, [], self.action)
            current = file_hash(self.root / 'segments/S01/v001/scene.html')
            requests = {
                'S01': {'local_source': {'path': 'author.html', 'hash': current}, 'reason': 'old strict reason'},
                'S02': {'local_patch': {'path': 'patch.json', 'hash': 'patch', 'source_hash': current},
                        'keyframe_recheck': {'reason': 'old review protocol'}},
                'S03': {'local_patch': {'path': 'patch.json', 'hash': 'patch', 'source_hash': 'older scene'}},
                'S04': {'temporal_recheck': {'reason': 'strict timing'}}}
            project.data['repair_requests'] = requests
        with Project(self.root, self.config, resume=True, review_strictness='balanced') as project:
            self.assertEqual(project.data['policy_history'][0]['repair_requests'], requests)
            self.assertEqual(project.data['repair_requests'], {'S02': {'local_patch': requests['S02']['local_patch']}})
            self.assertEqual(project.data['policy_history'][0]['preserved_local_repairs'],
                             project.data['repair_requests'])

    def test_returning_to_earlier_profile_creates_another_candidate(self):
        for index, profile in enumerate(('strict', 'balanced', 'strict')):
            with Project(self.root, self.config, resume=bool(index), review_strictness=profile) as project:
                result = project.task('segments/S01', {}, [], self.action)
                self.assertEqual(result['version'], index + 1)
                self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), index + 1)

    def test_policy_change_does_not_reset_candidate_limit(self):
        with Project(self.root, self.config, review_strictness='strict') as project:
            project.task('segments/S01', {}, [], self.action, max_attempts=1)
        with Project(self.root, self.config, resume=True, review_strictness='balanced') as project:
            with self.assertRaises(Blocked):
                project.task('segments/S01', {}, [], self.action, max_attempts=1)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)


class ReviewConfigCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'story.txt'
        self.source.write_text('A fish swims home.', encoding='utf-8')

    def generate_options(self, arguments=(), environment=None, dotenv=None):
        argv = ['video_agent', 'generate', str(self.source), '--out', str(self.root / 'out'), *arguments]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, environment or {}, clear=True), \
                patch.object(video_agent, 'dotenv_values', return_value=dotenv or {}), \
                patch.object(video_agent.shutil, 'which', return_value='/mock/ffmpeg'), \
                patch.object(video_agent, 'generate') as generate, patch('sys.stdout', io.StringIO()):
            video_agent.main()
            return generate.call_args.kwargs

    def test_generate_cli_environment_dotenv_precedence(self):
        environment = {'VIDEO_REVIEW_STRICTNESS': 'strict', 'VIDEO_RENDER_CONSISTENCY': 'tolerant'}
        dotenv = {'VIDEO_REVIEW_STRICTNESS': 'relaxed', 'VIDEO_RENDER_CONSISTENCY': 'exact',
                  'DEEPSEEK_API_KEY': 'test-only-not-a-secret'}
        self.assertEqual(self.generate_options(('--review-strictness', 'balanced'), environment, dotenv),
                         {'review_strictness': 'balanced', 'render_consistency': 'tolerant'})
        self.assertEqual(self.generate_options(environment=environment, dotenv=dotenv),
                         {'review_strictness': 'strict', 'render_consistency': 'tolerant'})
        self.assertEqual(self.generate_options(dotenv=dotenv),
                         {'review_strictness': 'relaxed', 'render_consistency': 'exact'})
        self.assertEqual(self.generate_options(('--resume',)),
                         {'review_strictness': None, 'render_consistency': None})

    def test_render_worker_never_loads_dotenv_and_forwards_mode(self):
        argv = ['video_agent', 'render', str(self.source), '--out', str(self.root / 'out'),
                '--worker', '--preview', '--render-consistency', 'tolerant']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {}, clear=True), \
                patch.object(video_agent, 'dotenv_values', side_effect=AssertionError('worker read .env')), \
                patch.object(video_agent.rendering, 'render') as render:
            video_agent.main()
            self.assertEqual(render.call_args.kwargs, {'consistency': 'tolerant'})
            self.assertNotIn('DEEPSEEK_API_KEY', os.environ)

    def test_invalid_environment_profile_is_rejected(self):
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as error:
            self.generate_options(environment={'VIDEO_REVIEW_STRICTNESS': 'disable'})
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
