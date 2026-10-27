"""隔离三方合成四元数与时间轴修复；不替代正式合并/清单/全套验收。"""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--quaternion-ref",default="3d8e7ed0884f9f91487af9cd03434268c0e92e92")
    parser.add_argument("--clock-ref",default="f411d1e0a73fbadf36e0b5b315b9e0b1421bc240")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.repo_root.resolve()
    def git(*values):return subprocess.check_output(["git",*values],cwd=root)
    refs=[git("rev-parse","--verify",r+"^{commit}").decode().strip() for r in (args.quaternion_ref,args.clock_ref)]
    base=git("merge-base",*refs).decode().strip()
    path="lingshu/world/world_model.py"
    assert all(git("diff","--name-only",base,r,"--","lingshu").decode().splitlines()==[path] for r in refs)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sandbox=Path(tempfile.mkdtemp(prefix="owned-composition-",dir=output)).resolve()
    assert sandbox.is_relative_to(output) and sandbox!=output
    try:
        with zipfile.ZipFile(io.BytesIO(git("archive","--format=zip",refs[0],"lingshu"))) as archive:
            assert all((sandbox/item.filename).resolve().is_relative_to(sandbox) for item in archive.infolist())
            archive.extractall(sandbox)
        inputs=[]
        for name,revision in (("quaternion",refs[0]),("base",base),("clock",refs[1])):
            item=sandbox/(name+".py")
            item.write_bytes(git("show",revision+":"+path).replace(b"\r\n",b"\n"))
            inputs.append(str(item))
        merged=subprocess.run(["git","merge-file","-p",*inputs],cwd=root,capture_output=True)
        assert merged.returncode==0,"源文件三方合并必须无冲突"
        (sandbox/path).write_bytes(merged.stdout)
        guards=["tests/test_world_rotation_quaternion_validity.py","tests/test_world_rotation_tick_clock.py"]
        (sandbox/"tests").mkdir()
        for guard,revision in zip(guards,refs):
            (sandbox/guard).write_bytes(git("show",revision+":"+guard).replace(b"\r\n",b"\n"))
        env={**os.environ,"PYTHONPATH":str(sandbox),"PYTHONDONTWRITEBYTECODE":"1"}
        if os.name!="nt":env["PYTEST_ADDOPTS"]=env.get("PYTEST_ADDOPTS","")+" --capture=no"
        result=subprocess.run([sys.executable,"-B","-X","utf8","-m","pytest",*guards,"-q","--tb=short",
            "--disable-warnings","-p","no:cacheprovider","--basetemp="+str(sandbox/"temp"),
            "--junitxml="+str(output/"composition.xml")],cwd=sandbox,env=env,capture_output=True,timeout=180)
        raw=(result.stdout+result.stderr).decode("utf-8",errors="replace").replace("\r\n","\n")
        (output/"composition.log").write_text(raw,encoding="utf-8",newline="\n")
        cases=ET.parse(output/"composition.xml").findall(".//testcase")
        failed=[t.get("name") for t in cases if t.find("failure") is not None or t.find("error") is not None]
        skipped=[t.get("name") for t in cases if t.find("skipped") is not None]
        hashes={p:hashlib.sha256((sandbox/p).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in [path,*guards]}
    finally:
        assert sandbox.is_relative_to(output) and sandbox!=output
        shutil.rmtree(sandbox)
    summary={"revisions":refs,"base":base,"merge_returncode":merged.returncode,"returncode":result.returncode,
        "tests":len(cases),"failures":failed,"skipped":skipped,"sha256_lf":hashes,
        "helper_sha256_lf":hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
        "owned_temp_cleaned":not sandbox.exists(),"scope":"隔离源码组合；非正式 Git/清单/全套验收"}
    (output/"composition.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if result.returncode==0 and len(cases)==45 and not failed and not skipped else 1


if __name__=="__main__":raise SystemExit(main())
