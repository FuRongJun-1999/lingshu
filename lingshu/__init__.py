"""灵枢（lingshu）· 聚合导出包 —— 世界模型 / 存算一体 / 自研神经网络 / 图像生成。"""
# issue #156 v2：**最早落点**——在本包任何子模块（含 core.py:17 `import json`）
# 的任何裸名 import 之前，消除 cwd/空串 sys.path 解析面。Python 语义保证
# `import lingshu.core.core` 必先执行本文件。见 lingshu/_pathguard.py（含精确
# 范围声明与逃生口 LINGSHU_ALLOW_CWD_IMPORTS）。
from . import _pathguard as _pathguard  # noqa: F401  (import 即执行，勿删)
