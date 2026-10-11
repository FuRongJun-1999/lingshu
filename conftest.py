# -*- coding: utf-8 -*-
"""仓库根注入 sys.path——测试件无需安装包即可直跑（python -m pytest tests/）。
各测试件头部原有的 sys.path.insert 保持不变（脚本模式独立可跑）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
