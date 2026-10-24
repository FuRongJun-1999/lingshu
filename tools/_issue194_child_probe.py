# -*- coding: utf-8 -*-
"""#194 守卫的子进程探针（不是测试文件，供 test_issue194_*.py 调用）。

为什么单独成文件：先前把这段代码内联成单行字符串传 `python -c`，
`try: ...; except ...:` 不能共处一个逻辑行 ⇒ SyntaxError，
守卫于是在**回滚源码后也「通过」**（假通过）。落成真文件就没有这个坑。

为什么主体收进 `main()`（#937）：本文件在 `tests/` 下，`pytest tests/` 的**收集期
会 import 它**；模块级 `sys.exit()` 会在收集期直接终止解释器（INTERNALERROR），
把整批测试带崩。上游守卫 `test_no_module_level_sys_exit_in_tests` 因此点名本文件。
收进 `main()` 后：**import 期零副作用**，而作为子进程直接跑时行为逐字不变
（`if __name__ == "__main__"` 块仍调用同一段逻辑、仍 exit 0）。

用法: python _issue194_child_probe.py <n_objects> <seed>
输出: NO_ERROR | RAISED_VALUEERROR | RAISED_OTHER <Type>
始终 exit 0（「是否抛错」由 stdout 审判词判定，rc 只表示脚本跑没跑完）。
"""
import os
import sys

ROOT = os.environ.get("LINGSHU_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from lingshu.nn import hex_text as ht  # noqa: E402


def main(argv=None) -> int:
    """跑一次探针；始终返回 0（判定看 stdout 审判词，不看 rc）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    n = int(args[0])
    seed = int(args[1])

    try:
        ht.make_multimodal_scene(np.random.default_rng(seed), 16, n)
        print("NO_ERROR")
    except ValueError as e:
        print("RAISED_VALUEERROR")
        print("MSG: %s" % (e,))
    except Exception as e:                      # noqa: BLE001
        print("RAISED_OTHER %s" % type(e).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
