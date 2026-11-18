"""终审覆盖、像素证据来源与发布门槛；替身不证明实际画质。"""
import base64
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model, prompts, rendering, workflow
from agent.common import read_json, write_json
from agent.store import Blocked, Project, Rejected
from fixtures import BOARD, FACTS, REQ, STORY, FakeModel


class FinalSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'out'
        self.config = {'story': STORY, 'seconds': 10, 'fps': 24}
        self.facts = dict(copy.deepcopy(FACTS), events=copy.deepcopy(FACTS['events'][:1]))
        self.board = {'shots': [dict(copy.deepcopy(BOARD['shots'][0]), end_frame=240)]}
        self.plan = {'shot_id': 'S01', 'keyframes': [
            {'frame_id': f'K{i + 1:02d}', 'frame_index': frame, 'state': '零件逐步拼合',
             'covers': ['R01'], 'requirements': copy.deepcopy(REQ)}
            for i, frame in enumerate([0, 48, 96, 168, 239])],
            'critical_frames': [24, 48, 72, 96, 132, 167, 168, 192, 238],
            'intervals': [
                {'start_frame': 0, 'end_frame': 6, 'method': 'hold'},
                {'start_frame': 6, 'end_frame': 168, 'method': 'motion'},
                {'start_frame': 168, 'end_frame': 240, 'method': 'hold'}],
            'allowed_black_ranges': []}
        self.actual = [0, 1, 2, 5, 6, 22, 23, 24, 25, 26, 34, 46, 47, 48, 49, 50,
                       68, 70, 71, 72, 73, 74, 86, 94, 95, 96, 97, 98, 102, 127,
                       130, 131, 132, 133, 134, 137, 165, 166, 167, 168, 169, 170,
                       171, 190, 191, 192, 193, 194, 204, 205, 236, 237, 238, 239]

    def seed(self, project):
        def segment(directory, version, previous):
            video = directory / 'video.mp4'
            video.write_bytes(b'encoded-shot-placeholder')
            samples = []
            for frame in self.actual:
                path = directory / f'source_{frame}.png'
                path.write_bytes(b'uncoded-source-placeholder')
                samples.append({'frame_index': frame, 'path': str(path.relative_to(self.root))})
            return {'shot_id': 'S01', 'start_frame': 0, 'end_frame': 240,
                    'video': str(video.relative_to(self.root)), 'frame_plan': self.plan,
                    'samples': samples}
        shot = project.task('segments/S01', {}, [], segment)
        project.task('checks/timeline', {}, ['segments/S01'],
                     lambda *args: {'status': 'PASS'})
        return [shot]

    @staticmethod
    def sample_video(video, directory, indices):
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for number, frame in enumerate(indices):
            name = f'preview_{number:02d}.png'
            (directory / name).write_bytes(f'decoded-final-frame:{frame}'.encode())
            rows.append({'frame_index': frame, 'path': name})
        write_json(directory / 'preview_manifest.json', rows)
        return rendering.preview_images(directory)

    def renderers(self):
        stack = ExitStack()
        stack.enter_context(patch.object(rendering, 'encode_segments',
            side_effect=lambda paths, output: output.write_bytes(b'encoded-final-placeholder')))
        stack.enter_context(patch.object(rendering, 'inspect_video', return_value={'passed': True}))
        self.sample = stack.enter_context(patch.object(rendering, 'sample_video',
                                                      side_effect=self.sample_video))
        return stack

    def test_interval_interior_and_existing_frames_are_decoded_from_final_video(self):
        fake, requests = FakeModel(), []

        def ask(messages, **kwargs):
            requests.append(messages)
            return fake(messages)

        with Project(self.root, self.config) as project:
            shots = self.seed(project)
            with self.renderers(), patch.object(model, 'ask', side_effect=ask):
                workflow.final_check(project, self.facts, self.board, shots)
        directory = self.root / 'exports/final/v001'
        video, frames, indices = self.sample.call_args.args
        self.assertEqual(video, directory / 'candidate_final.mp4')
        self.assertEqual(frames, directory / 'frames')
        # 此夹具按旧外观取样规则选中的中帧为 98，不能因代表帧限额丢弃。
        self.assertTrue({0, 48, 96, 98, 168, 239} <= set(indices))
        self.assertTrue(any(96 < frame < 168 for frame in indices))
        self.assertTrue(set(indices) <= set(self.actual))
        sampling = read_json(directory / 'final_sampling.json')
        self.assertEqual(sampling['final_evidence_protocol'], 'temporal_coverage_v2')
        self.assertEqual(sampling['decoded_global_frame_indices'], indices)
        selected = sampling['shots'][0]['representative_local_frame_indices']
        self.assertLessEqual(len(selected), 16)
        self.assertEqual(sampling['shots'][0]['actual_local_frame_indices'], self.actual)
        request = next(messages for messages in requests
                       if json.loads(messages[1]['content'][0]['text']).get('target_id') == 'E01')
        image_rows = [row for row in request[1]['content'] if row['type'] == 'image_url']
        decoded = [base64.b64decode(row['image_url']['url'].split(',', 1)[1])
                   for row in image_rows]
        self.assertEqual(decoded, [f'decoded-final-frame:{frame}'.encode() for frame in indices])

    def test_event_is_blind_to_observation_prose_but_story_review_keeps_it(self):
        fake = FakeModel()
        marker = 'diagnostic-only-observer-claims-all-motion-passed'

        def ask(messages, **kwargs):
            result = fake(messages)
            if messages[0]['content'] == prompts.OBSERVE:
                for row in result['observations']:
                    row['description'] = marker
            return result

        with Project(self.root, self.config) as project:
            shots = self.seed(project)
            with self.renderers(), patch.object(model, 'ask', side_effect=ask):
                workflow.final_check(project, self.facts, self.board, shots)
        event = next(payload for _, payload in fake.calls if payload.get('target_id') == 'E01')
        self.assertNotIn('independent_observations', event)
        self.assertNotIn(marker, json.dumps(event))
        self.assertEqual(event['evidence_mode'], 'pixels')
        self.assertEqual(event['final_evidence_protocol'], 'temporal_coverage_v2')
        self.assertIn('pixel_evidence_role', event)
        self.assertEqual(event['requirements'][0],
                         {'id': 'E01', 'hard': True, 'text': self.facts['events'][0]['must_show']})
        story = next(payload for _, payload in fake.calls if payload.get('target_id') == 'final_story')
        self.assertEqual(story['evidence_mode'], 'observations')
        self.assertIn(marker, json.dumps(story['independent_observations']))
        saved = read_json(self.root / 'exports/final/v001/independent_observations.json')
        self.assertTrue(saved and all(row['description'] == marker for row in saved))

    def test_event_failure_still_blocks_publication(self):
        fake = FakeModel()

        def ask(messages, **kwargs):
            result = fake(messages)
            if result.get('target_id') == 'E01':
                result['checks'][0]['result'] = 'FAIL'
                result['repair']['shot_ids'] = ['S01']
            return result

        with Project(self.root, self.config) as project:
            shots = self.seed(project)
            with self.renderers(), patch.object(model, 'ask', side_effect=ask):
                with self.assertRaises(Rejected):
                    workflow.final_check(project, self.facts, self.board, shots)
            with self.assertRaises(Blocked):
                project.publish(self.root / 'exports/final/v001/candidate_final.mp4')
            self.assertEqual(project.data['tasks']['exports/final']['status'], 'FAIL')
        self.assertFalse((self.root / 'final.mp4').exists())
        self.assertFalse(any(payload.get('target_id') == 'final_story' for _, payload in fake.calls))

    def test_new_evidence_protocol_invalidates_legacy_final_task(self):
        with Project(self.root, self.config) as project:
            shots = self.seed(project)

            def legacy(directory, version, previous):
                video = directory / 'candidate_final.mp4'
                video.write_bytes(b'legacy-final')
                return {'status': 'PASS', 'video': str(video.relative_to(self.root))}

            project.task('exports/final', {'story': self.facts, 'storyboard': self.board},
                         ['checks/timeline', 'segments/S01'], legacy)
            with self.renderers(), patch.object(model, 'ask', side_effect=FakeModel()):
                workflow.final_check(project, self.facts, self.board, shots)
            self.assertEqual(project.data['tasks']['exports/final']['version'], 2)
            self.assertEqual(len(project.data['tasks']['exports/final']['attempts']), 2)
        self.assertEqual((self.root / 'exports/final/v001/candidate_final.mp4').read_bytes(), b'legacy-final')
        self.assertEqual(self.sample.call_args.args[0],
                         self.root / 'exports/final/v002/candidate_final.mp4')


if __name__ == '__main__':
    unittest.main()
