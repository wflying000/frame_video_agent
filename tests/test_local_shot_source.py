"""首候选本地镜头源码不能跳过像素、时序、技术审核或绕过版本绑定。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import drawing, model, prompts, rendering
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FACTS, HTML, STORY, VISUAL, FakeModel, frames
import test_workflow

class LocalShotSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'
        self.config = {'story': STORY, 'fps': 2}

    def seed(self, project, shot_id='S01'):
        project.task('plans/story', {}, [], lambda *args: FACTS)
        project.task('plans/visual', {}, ['plans/story'], lambda *args: VISUAL)
        project.task('plans/storyboard', {}, ['plans/story', 'plans/visual'], lambda *args: BOARD)
        def asset(directory, *_):
            image = directory / 'reference.png'
            image.write_bytes(b'actual reference pixels')
            return {'version': 1, 'image': str(image.relative_to(project.root))}
        reference = project.task('assets/P01', {}, ['plans/visual'], asset)
        project.task(f'plans/frames/{shot_id}', {}, ['plans/storyboard'], lambda *args: frames(shot_id))
        source = project.root / 'authored.html'
        source.write_text(HTML)
        descriptor = {'path': 'authored.html', 'hash': file_hash(source)}
        project.data['repair_requests'][shot_id] = {'local_source': descriptor}
        return {'P01': reference}, descriptor

    def test_local_code_skips_only_generation_and_preserves_continuity(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}) as inspect:
            refs, descriptor = self.seed(project, 'S02')
            prior_image = self.root / 'prior.png'
            prior_image.write_bytes(b'previous actual frame')
            previous = {'shot_id': 'S01', 'version': 1, 'last_frame': 'prior.png',
                        'start_frame': 0, 'end_frame': 4, 'html': 'authored.html'}
            frame_plan = frames('S02')
            frame_plan['keyframes'][0]['requirements'] = [
                {'id': 'R01', 'hard': True, 'text': '首帧承接上一镜头状态'}]
            result = drawing.shot(project, BOARD['shots'][1], frame_plan, refs, previous_shot=previous)
            folder = self.root / 'segments/S02/v001'
            self.assertEqual(result['version'], 1)
            self.assertEqual([system for system, _ in fake.calls],
                [prompts.OBSERVE, prompts.REVIEW, prompts.OBSERVE, prompts.REVIEW, prompts.REVIEW, prompts.REVIEW])
            reviews = [data for system, data in fake.calls if system == prompts.REVIEW]
            observations = [data for system, data in fake.calls if system == prompts.OBSERVE]
            self.assertEqual([data['frame_indices'] for data in observations], [[0], [0, 3]])
            self.assertEqual([data['target_id'] for data in reviews], ['S02_K01', 'S02_K02', 'S02', 'S02'])
            self.assertTrue(all(data['evidence_mode'] == 'pixels' for data in reviews))
            self.assertIn('DEPTH_VOLUME', reviews[0]['required_check_ids'])
            self.assertIn('DEPTH_STABILITY', reviews[-1]['required_check_ids'])
            self.assertIn('entry_reference', reviews[0])
            self.assertEqual(read_json(folder / 'continuity_reference.json')['image_hash'], file_hash(prior_image))
            self.assertEqual((folder / 'scene.html').read_text(), HTML)
            proof = read_json(folder / 'local_source_provenance.json')
            self.assertEqual(proof, {**descriptor, 'provenance': 'local_code_authoring', 'requires_full_review': True})
            self.assertFalse((folder / 'generated.json').exists())
            inspect.assert_called_once()
            self.assertTrue((folder / 'technical_check.json').exists())

    def test_failed_pixel_check_does_not_pass_or_export(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            result = fake(messages)
            if result.get('target_id') == 'S01_K01':
                result['status'] = 'FAIL'
                check = next(row for row in result['checks'] if row['requirement_id'] == 'DEPTH_VOLUME')
                check.update(result='FAIL', evidence='主体仍扁平')
            return result
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video') as inspect:
            refs, _ = self.seed(project)
            with self.assertRaises(Blocked):
                drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            folder = self.root / 'segments/S01/v001'
            self.assertEqual(read_json(folder / 'failure.json')['status'], 'FAIL')
            self.assertFalse((folder / 'result.json').exists())
            self.assertFalse((folder / 'preview/video.mp4').exists())
            self.assertFalse((self.root / 'final.mp4').exists())
            inspect.assert_not_called()

    def test_outside_hash_and_existing_candidate_are_rejected_before_calls(self):
        for fault in ('outside', 'symlink', 'hash', 'previous'):
            with self.subTest(fault=fault), Project(self.root / fault, self.config) as project, \
                    patch.object(model, 'ask', side_effect=AssertionError('应在模型调用前拒绝')):
                refs, descriptor = self.seed(project)
                if fault in ('outside', 'symlink'):
                    outside = project.root.parent / 'outside.html'
                    outside.write_text(HTML)
                    if fault == 'outside':
                        descriptor['path'] = '../outside.html'
                    else:
                        (project.root / 'link.html').symlink_to(outside)
                        descriptor['path'] = 'link.html'
                    descriptor['hash'] = file_hash(outside)
                elif fault == 'hash':
                    descriptor['hash'] = '0' * 64
                else:
                    def fail(*_):
                        raise ValueError('previous failure')
                    from agent.store import Rejected
                    with self.assertRaises(Rejected):
                        project.task('segments/S01', {'old': True}, [], fail, retry=False)
                with self.assertRaises(Blocked):
                    drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
                self.assertEqual(project.data['calls'], 0)

    def test_local_source_and_patch_cannot_be_combined(self):
        with Project(self.root, self.config) as project:
            refs, _ = self.seed(project)
            project.data['repair_requests']['S01']['local_patch'] = {'path': 'some-patch.json'}
            with self.assertRaisesRegex(ValueError, '不能同时指定'):
                drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            self.assertNotIn('segments/S01', project.data['tasks'])

if __name__ == '__main__':
    unittest.main()
