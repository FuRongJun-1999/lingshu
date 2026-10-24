"""隔离组合多份世界模型增量的源码与守卫，不替代正式 Git/清单/全套验收。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revisions', nargs=3, required=True,
                        help='依次为 W3 试演、W2 身份、W1 反馈的固定提交')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    refs = [git('rev-parse', '--verify', r + '^{commit}').decode().strip() for r in args.revisions]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sandbox = Path(tempfile.mkdtemp(prefix='owned-composition-', dir=output)).resolve()
    assert sandbox.is_relative_to(output) and sandbox != output
    merges, hashes = [], {}
    try:
        archive = git('archive', '--format=zip', refs[0], 'lingshu')
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            assert all((sandbox / p.filename).resolve().is_relative_to(sandbox) for p in files.infolist())
            files.extractall(sandbox)
        for revision in refs[1:]:
            base = git('merge-base', refs[0], revision).decode().strip()
            changed = git('diff', '--name-only', base, revision, '--', 'lingshu').decode().splitlines()
            for path in changed:
                target = (sandbox / path).resolve()
                assert target.is_relative_to(sandbox)
                other = git('show', revision + ':' + path)
                old = subprocess.run(['git', 'show', base + ':' + path], cwd=ROOT, capture_output=True)
                if old.returncode:
                    assert not target.exists() or target.read_bytes() == other
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(other)
                else:
                    assert target.exists()
                    inputs = []
                    # Windows 的 git archive 可能输出 CRLF，git show 为 LF；
                    # 三方输入统一为 Git 源码口径，避免把换行差异当作整文件修改。
                    for name, data in [('current', target.read_bytes()), ('base', old.stdout), ('other', other)]:
                        item = sandbox / (name + '.py')
                        item.write_bytes(data.replace(b'\r\n', b'\n'))
                        inputs.append(str(item))
                    merged = subprocess.run(['git', 'merge-file', '-p', *inputs], cwd=ROOT, capture_output=True)
                    if merged.returncode:
                        (output / 'merge-failure.txt').write_bytes(merged.stdout + merged.stderr)
                    assert merged.returncode == 0, f'三方合并冲突：{path} ({revision})'
                    target.write_bytes(merged.stdout)
                    merges.append({'revision':revision, 'base':base, 'path':path, 'returncode':0})
        guards = ['tests/test_world_counterfactual_isolation.py',
                  'tests/test_world_identity_qualification.py', 'tests/test_world_qualified_feedback.py']
        (sandbox / 'tests').mkdir()
        for path, revision in zip(guards, refs):
            (sandbox / path).write_bytes(git('show', revision + ':' + path))
        joint = 'tests/test_joint_world_branches.py'
        (sandbox / joint).write_text('''from copy import deepcopy
import pickle
from lingshu.world.world_model import UnifiedWorldModel

def test_conservative_identity_audit_survives_counterfactual():
    model = UnifiedWorldModel(size=80, identity_policy="conservative")
    for x in range(6, 12):
        model.perceive([{"eid":"actor","category":"person","pos":[x,1.5,2]},
                        {"eid":"goal","category":"marker","pos":[30,1.5,2]}])
    before = pickle.dumps(model.__dict__, protocol=4)
    result = model.counterfactual(relations=[{"source":"actor","relation":"flee","target":"goal"}])
    assert result["alternative"]["predictions"]["actor"]["predicted"][0] == 10
    assert pickle.dumps(model.__dict__, protocol=4) == before
    assert model.identity_policy == "conservative"
''', encoding='utf-8')
        guards.append(joint)
        env = {**os.environ, 'PYTHONPATH':str(sandbox), 'PYTHONDONTWRITEBYTECODE':'1'}
        if os.name != 'nt':
            env['PYTEST_ADDOPTS'] = env.get('PYTEST_ADDOPTS', '') + ' --capture=no'
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'pytest', *guards,
            '-q', '--tb=short', '-p', 'no:cacheprovider', '--basetemp=' + str(sandbox / 'temp'),
            '--junitxml=' + str(output / 'composition.xml')], cwd=sandbox, env=env,
            capture_output=True, timeout=180)
        raw = (result.stdout + result.stderr).decode('utf-8', errors='replace').replace('\r\n', '\n')
        (output / 'composition.log').write_text(raw, encoding='utf-8')
        tests = ET.parse(output / 'composition.xml').findall('.//testcase')
        failures = [t.get('name') for t in tests if t.find('failure') is not None or t.find('error') is not None]
        skipped = [t.get('name') for t in tests if t.find('skipped') is not None]
        paths = {r['path'] for r in merges} | set(guards) | {'lingshu/world/identity.py'}
        hashes = {p: hashlib.sha256((sandbox / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}
    finally:
        assert sandbox.is_relative_to(output) and sandbox != output
        shutil.rmtree(sandbox)
    summary = {'revisions':refs, 'merges':merges, 'returncode':result.returncode,
        'tests':len(tests), 'failures':failures, 'skipped':skipped, 'sha256_lf':hashes,
        'owned_temp_cleaned':not sandbox.exists(), 'scope':'仅隔离源码组合，非正式 Git/清单/全套验收'}
    (output / 'composition.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result.returncode == 0 and not failures and not skipped else 1

if __name__ == '__main__':
    raise SystemExit(main())
