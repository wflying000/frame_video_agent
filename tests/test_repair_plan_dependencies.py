"""已批准计划依赖变化时，不让旧候选局部计划覆盖新计划。"""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from agent import repair
from agent.common import file_hash, read_json, write_json
from fixtures import BOARD, frames

class RepairPlanDependencyTests(unittest.TestCase):
    def setup_case(self, root, old_versions, current_versions):
        previous = {'version': 1, 'path': 'segments/S01/v001'}
        base = root / previous['path']
        base.mkdir(parents=True)
        source = base / 'scene.html'
        source.write_text('<html>keep the actual patched scene source</html>')
        old_plan = frames('S01')
        old_plan['keyframes'][1]['state'] = '旧候选局部重规划后的状态'
        latest = frames('S01')
        latest['keyframes'][1]['state'] = '最新已批准计划的状态'
        saved = {'frame_plan': old_plan}
        if old_versions is not None:
            saved['input_versions'] = old_versions
        write_json(base / 'frame_tasks.json', saved)
        node = {'attempts': [previous]}
        if current_versions is not None:
            node['input_versions'] = current_versions
        project = SimpleNamespace(root=root, data={'tasks': {'segments/S01': node}})
        return project, previous, source, old_plan, latest

    def test_changed_storyboard_or_frame_dependency_uses_latest_plan_without_old_regression(self):
        versions = {'plans/storyboard': {'version': 1, 'output_hash': 'old-board'},
                    'plans/frames/S01': {'version': 2, 'output_hash': 'old-plan'}}
        for changed_key in versions:
            with self.subTest(changed=changed_key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                current = deepcopy(versions)
                current[changed_key] = {'version': 3, 'output_hash': 'new-approved'}
                project, previous, source, old_plan, latest = self.setup_case(root, versions, current)
                original_hash = file_hash(source)
                original_plan_hash = file_hash(source.parent / 'frame_tasks.json')
                with patch.object(repair, 'regression_context', side_effect=AssertionError('不能恢复旧计划约束')):
                    actual_source, plan, strategy = repair.patch_context(project, root / 'segments/S01/v002',
                        BOARD['shots'][0], latest, previous)
                self.assertEqual(plan, latest)
                self.assertNotEqual(plan, old_plan)
                self.assertEqual(actual_source, source)
                self.assertEqual(file_hash(source), original_hash)
                self.assertEqual(file_hash(source.parent / 'frame_tasks.json'), original_plan_hash)
                self.assertNotIn('regression', strategy)
                update = strategy['updated_plan']
                self.assertEqual(update['frame_plan_task'], 'plans/frames/S01')
                self.assertEqual(list(update['changed_dependencies']), [changed_key])
                self.assertEqual(read_json(root / 'segments/S01/v002/repair_strategy.json')['updated_plan'], update)

    def test_unchanged_or_legacy_dependencies_keep_effective_local_plan_and_source(self):
        versions = {'plans/storyboard': {'version': 1, 'output_hash': 'board'},
                    'plans/frames/S01': {'version': 2, 'output_hash': 'plan'}}
        for old_versions, current in ((versions, deepcopy(versions)), (None, versions), (versions, None)):
            with self.subTest(old=old_versions, current=current), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project, previous, source, old_plan, latest = self.setup_case(root, old_versions, current)
                original_hash = file_hash(source)
                with patch.object(repair, 'regression_context', return_value={'verified': 'old pixels'}) as regression:
                    actual_source, plan, strategy = repair.patch_context(project, root / 'segments/S01/v002',
                        BOARD['shots'][0], latest, previous)
                self.assertEqual(plan, old_plan)
                self.assertEqual(actual_source, source)
                self.assertEqual(file_hash(source), original_hash)
                self.assertNotIn('updated_plan', strategy)
                self.assertEqual(strategy['regression'], {'verified': 'old pixels'})
                regression.assert_called_once()

if __name__ == '__main__':
    unittest.main()
