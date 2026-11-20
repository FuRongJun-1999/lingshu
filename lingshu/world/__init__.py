"""lingshu.world —— 见本目录 README.md。

本包的 register_world_process 把 world_process、motion_groups 与 process_ops 三个模块
记到引擎实例（依赖方向 world → core）。导入本包只带入模块定义：不注册、不改引擎状态。
"""
from __future__ import annotations

from importlib import import_module

__all__ = ["register_world_process", "unproject_depth_samples"]


def __getattr__(name):
    if name == "unproject_depth_samples":
        return getattr(import_module(".depth_samples", __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))


def register_world_process(engine) -> None:
    """把 world_process、motion_groups 与 process_ops 记到**该引擎实例**。

    world_process 承载时间会意、示教迁移与事件回声；motion_groups 承载运动组合体；
    process_ops 承载过程补全与过程重组（第 2 组）。

    用法（新建或重开引擎后各调一次）：

        from lingshu.world import register_world_process

        engine = SpacetimeMemoryEngine(db_path="store.db")
        register_world_process(engine)
        engine.world_process("recognize", {})

    core 不导入 world：未注册的引擎调用 engine.world_process 时返回
    {"status": "error", ...}，error 里写明上面的初始化步骤。
    本函数只在传入的实例上写注入槽，不影响其它实例、不产生模块级全局状态；
    可对同一实例重复调用（幂等）。
    """
    from . import motion_groups, process_ops, world_process
    engine.register_world_process(world_process, motion_groups, process_ops)
