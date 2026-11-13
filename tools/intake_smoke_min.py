# -*- coding: utf-8 -*-
"""本仓内可执行的 intake 冒烟复跑（issue #80 · #8 的等价最小脚本）。

背景
----
`docs/intake/body-export-v0.2/README.md` 把「模块 import 冒烟 ok=51」列为导出面的
实证读数，但原复跑命令是 `python <AEIS>/tools/public_export/smoke.py <本仓根>`——
`<AEIS>` 是**未定义的占位符**，且该工具**不随本仓导出**，读者在本仓内无法执行。
这与同文档 §一「结论均以能被独立复跑为准」直接冲突（issue #80 的 #8 条）。

本脚本只做「等价的最小面」：**在本仓内**发现并 import 全部 `lingshu.*` 模块，
外加一次构造性运行（圆掩膜 → `image_to_grid`），打印 ok/fail 计数。
它**不**等价于上游 smoke.py 的全部检查（见下方「不做项」），只保证「模块可 import
+ 核心路径可跑通」这两件在本仓内可独立复跑。

用法
----
    python tools/intake_smoke_min.py            # 全量：发现 + import + 构造性运行
    python tools/intake_smoke_min.py --json     # 机器可读
    python tools/intake_smoke_min.py --quiet    # 只打印汇总行

退出码
------
0 = 全部 import 成功且构造性运行通过；1 = 有失败（逐条打印）。

不做项（与上游 smoke.py 的差距，如实声明）
------------------------------------------
· 不校验导出副本与源的逐位一致（那是 `tools/verify_manifest.py` 的面）
· 不做分割读回、不做脱敏字段比对
· 不覆盖语料依赖测试件
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import sys


def discover() -> list:
    """发现本仓 `lingshu` 包下全部模块（不含测试件）。"""
    import lingshu
    return sorted(m.name for m in pkgutil.walk_packages(lingshu.__path__, prefix="lingshu."))


def smoke() -> dict:
    names = discover()
    ok, fail = 0, []
    for name in names:
        try:
            importlib.import_module(name)
            ok += 1
        except BaseException as exc:  # noqa: BLE001
            fail.append({"module": name, "error": type(exc).__name__, "message": str(exc)[:200]})

    constructive = {"ok": False, "detail": ""}
    try:
        import numpy as np
        from lingshu.nn.hex_cnn import image_to_grid

        n = 48
        yy, xx = np.mgrid[0:n, 0:n]
        c = (n - 1) / 2.0
        r = n / 3.0
        mask = ((yy - c) ** 2 + (xx - c) ** 2) <= r * r
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[mask] = (255, 255, 255)
        feats, _ = image_to_grid(img, cells_across=16)
        constructive = {"ok": True,
                        "detail": "圆掩膜 %d px → image_to_grid %r" % (int(mask.sum()), feats.shape)}
    except BaseException as exc:  # noqa: BLE001
        constructive = {"ok": False, "detail": "%s: %s" % (type(exc).__name__, exc)}

    return {"discovered": len(names), "import_ok": ok, "import_fail": fail,
            "constructive": constructive,
            "passed": (len(fail) == 0 and constructive["ok"])}


def main() -> int:
    ap = argparse.ArgumentParser(description="本仓内 intake 冒烟复跑（issue #80 #8 等价最小面）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总行")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    res = smoke()

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.quiet:
        print("import ok=%d fail=%d constructive=%s"
              % (res["import_ok"], len(res["import_fail"]),
                 "ok" if res["constructive"]["ok"] else "FAIL"))
    else:
        print("发现模块 = %d" % res["discovered"])
        print("import 冒烟 = ok %d / fail %d" % (res["import_ok"], len(res["import_fail"])))
        for f in res["import_fail"]:
            print("  FAIL %-52s %s: %s" % (f["module"], f["error"], f["message"]))
        print("构造性运行 = %s  %s"
              % ("ok" if res["constructive"]["ok"] else "FAIL", res["constructive"]["detail"]))
        print("")
        print("PASS" if res["passed"] else "FAIL")

    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
