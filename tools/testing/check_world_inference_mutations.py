"""Named, single-failure mutations of the shared world inference contract.

Snapshots and logs are written under artifacts/; production files stay intact.
Run: python -X utf8 -B tools/testing/check_world_inference_mutations.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/world_inference_mutations'
TEST = 'tests/test_world_inference_window.py::'
MUTATIONS = [
    ('M1_dt', 'motion.py', 'dt = t1 - t0', 'dt = 1',
     TEST + 'test_observation_window_keeps_real_dt_and_joint_context[False]'),
    ('M2_empty_joint', 'motion.py', 'if not common:', 'if False:',
     TEST + 'test_window_with_no_joint_observation_does_not_read_old_frames'),
    ('M3_reverse_direction', 'motion.py', '(pb, previous[1], -tx, -tz, scores_b)',
     '(pb, previous[1], tx, tz, scores_b)', TEST + 'test_two_trajectories_have_independent_direction_scores'),
    ('M4_context', 'motion.py', 'if self.start and common != (1 << len(self.frames)) - 1:', 'if False:',
     TEST + 'test_observation_window_keeps_real_dt_and_joint_context[False]'),
    ('M5_stop_interval', 'motion.py', 'intervals += 1', 'intervals += int(pa != previous[0])',
     TEST + 'test_reverse_pairs_share_context_scan_and_count_stationary_observations'),
    ('M6_min_samples', 'world_model.py', 'if directional >= 3 and abs(c) > abs(best_c):',
     'if directional >= 1 and abs(c) > abs(best_c):',
     TEST + 'test_new_relation_requires_three_measured_displacements[UnifiedWorldModel]'),
    ('M7_stop_evidence', 'world_learner.py',
     '(measured >= 3 and n == 0 and motion[3] is None)', 'False',
     TEST + 'test_measured_stop_removes_relation_instead_of_preserving_it_as_stale[CuriosityExplorer]'),
    ('M8_mixed_evidence', 'world_learner.py',
     '(measured >= 3 and n == 0 and motion[3] is None)', '(measured >= 3)',
     TEST + 'test_mixed_sparse_evidence_does_not_erase_a_supported_learner_relation'),
    ('M9_joint_mask', 'motion.py', 'self.masks.get(a, 0) & self.masks.get(b, 0)',
     'self.masks.get(a, 0) | self.masks.get(b, 0)',
     TEST + 'test_window_with_no_joint_observation_does_not_read_old_frames'),
    ('M10_coincident', 'world_learner.py', ' and motion[3] is None', '',
     TEST + 'test_coincident_moving_entities_are_not_misclassified_as_a_measured_stop[CuriosityExplorer]'),
    ('M11_unified_mixed_evidence', 'world_model.py',
     'sufficient = (sufficient or directional >= 3\n'
     '                              or (samples >= 3 and directional == 0\n'
     '                                  and observations.motion(a)[3] is None))',
     'sufficient = sufficient or samples >= 3',
     'tests/test_pr460_review_relation_retention.py::'
     'test_unified_retains_candidate_until_supported_or_contradicted[one-direction]'),
    ('M12_unified_coincident', 'world_model.py',
     ' and observations.motion(a)[3] is None', '',
     TEST + 'test_coincident_moving_entities_are_not_misclassified_as_a_measured_stop[UnifiedWorldModel]'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, filename, old, new, nodeid in MUTATIONS:
        snapshot = OUT / name
        shutil.copytree(ROOT / 'lingshu', snapshot / 'lingshu', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        (snapshot / 'tests').mkdir(exist_ok=True)
        for test_filename in ('test_world_inference_window.py', 'test_pr460_review_relation_retention.py'):
            shutil.copyfile(ROOT / 'tests' / test_filename, snapshot / 'tests' / test_filename)
        source = snapshot / 'lingshu/world' / filename
        text = source.read_text(encoding='utf-8')
        assert text.count(old) == 1, (name, 'mutation target must be unique')
        source.write_text(text.replace(old, new, 1), encoding='utf-8')
        xml = OUT / (name + '.xml')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'pytest', nodeid, '-q',
                                 '-p', 'no:cacheprovider', '--rootdir', str(snapshot),
                                 '--confcutdir', str(snapshot), '--junitxml', str(xml)], cwd=snapshot,
                                env={**os.environ, 'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1'},
                                capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        log = result.stdout + result.stderr
        (OUT / (name + '.log')).write_text(log, encoding='utf-8')
        suite = ET.parse(xml).getroot().find('testsuite')
        failure = suite.find('testcase/failure')
        failure_message = failure.get('message', '') if failure is not None else ''
        # These lifecycle guards must fail at the business assertion, not [0].
        semantic_failure = (name not in {'M8_mixed_evidence', 'M10_coincident',
                                         'M11_unified_mixed_evidence', 'M12_unified_coincident'}
                            or ('AssertionError' in failure_message and 'IndexError' not in failure_message))
        killed = (semantic_failure and result.returncode == 1 and suite.get('tests') == '1' and suite.get('failures') == '1'
                  and suite.get('errors') == '0' and suite.get('skipped') == '0'
                  and 'FAILED ' + nodeid in log)
        records.append({'mutation': name, 'nodeid': nodeid, 'killed': killed, 'exit_code': result.returncode,
                        'failure_message': failure_message})
        print(name, 'KILLED' if killed else 'NOT_KILLED', nodeid, flush=True)
    (OUT / 'results.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f"SUMMARY {sum(r['killed'] for r in records)}/{len(records)}")
    return 0 if all(r['killed'] for r in records) else 1


if __name__ == '__main__':
    sys.exit(main())
