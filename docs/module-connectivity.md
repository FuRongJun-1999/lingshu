# 灵枢模块主链连通性实测台账（#172）

> **生成方式**：由 `python -X utf8 tools/module_topology_audit.py --write` 基于 AST 静态与惰性导入扫描机械生成，
> 并由 `--check` 门禁及 `tests/test_issue172_module_topology_audit.py` 守护（台账漂移即红）。

## 一、盘点口径与白箱定义

- **扫描范围**：`lingshu/` 下全部 `.py` 文件（共 **94** 个文件：**88** 个非 `__init__` 模块 + **6** 个 `__init__.py`）。
- **导入边识别**：AST 扫描 `import ...`、`from ... import ...`（含相对/绝对/子包解析）及 `importlib.import_module("lingshu....")` / `__import__("lingshu....")`（含 `core.py` 方法体内的惰性导入）。
- **主入口根节点**：`lingshu/core/core.py`（`SpacetimeMemoryEngine`）、`lingshu/world/world_model.py`（`UnifiedWorldModel`）、`lingshu/world/scene_model.py`（`WorldModel`）。
- **三分类定义**（对全部 **88** 个非 `__init__` 模块互斥完备划分）：
  1. `main_chain`（**核心主入口可达**）：从主入口根节点沿 `lingshu/` 内有向 import 边可达；
  2. `subgraph_only`（**包内子图互调**）：未从主入口直接可达，但在 `lingshu/` 内入度 > 0（含通过包 `__init__.py` 如 `lingshu/world/__init__.py::register_world_process` 等静态解析包内边 接入者）；
  3. `standalone_leaf`（**独立原型叶模块**）：`lingshu/` 包内入度 == 0（包内无任何模块导入它，仅被 `tests/`、`bench/`、`tools/` 或直接 CLI 消费）。

## 二、汇总读数（实测）

| 分类 | 模块数 | 占比 | 说明 |
|---|---:|---:|---|
| `main_chain`（核心主入口可达） | 41 | 46.6% | 由 `core.py`（含惰性边界）或 `WorldModel` / `UnifiedWorldModel` 静态/惰性可达 |
| `subgraph_only`（包内子图互调） | 19 | 21.6% | 仅在子包/扩展内部或经 `__init__.py`（含依赖倒置注册接口）互调 |
| `standalone_leaf`（独立原型叶模块） | 28 | 31.8% | 包内入度为 0，仅被测试 / 基准 / CLI 独立调用 |
| **非 `__init__` 合计** | **88** | **100.0%** | 与 `docs/architecture.md` 计数同口径 |

### 按层分布

| 层 | 非 `__init__` 数 | `main_chain` | `subgraph_only` | `standalone_leaf` |
|---|---:|---:|---:|---:|
| `root` | 1 | 1 | 0 | 0 |
| `core` | 15 | 8 | 3 | 4 |
| `world` | 47 | 30 | 6 | 11 |
| `nn` | 14 | 0 | 5 | 9 |
| `gen` | 7 | 0 | 5 | 2 |
| `vision` | 4 | 2 | 0 | 2 |

## 三、全量模块逐件台账（88 个非 `__init__` 模块）

| 模块路径 | 层 | 状态 | 主入口可达径 | 包内入度 / 导入方 | 包内出度 |
|---|---|---|---|---:|---:|
| `lingshu/_pathguard.py` | `root` | `main_chain`（核心主入口可达） | core.py | 3（`lingshu/__init__.py`, `lingshu/core/component_resolver.py`, `lingshu/core/core.py`） | 0 |
| `lingshu/core/activation.py` | `core` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/core/component_resolver.py` | `core` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 1 |
| `lingshu/core/condition_normalize.py` | `core` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/core/verdict.py`） | 0 |
| `lingshu/core/contradiction.py` | `core` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 1 |
| `lingshu/core/core.py` | `core` | `main_chain`（核心主入口可达） | core.py | 3（`lingshu/core/longterm_gate.py`, `lingshu/core/structure_service.py`, `lingshu/core/structure_writer.py`） | 21 |
| `lingshu/core/evidence.py` | `core` | `main_chain`（核心主入口可达） | core.py | 2（`lingshu/core/core.py`, `lingshu/core/tms.py`） | 0 |
| `lingshu/core/longterm_gate.py` | `core` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 1 |
| `lingshu/core/provenance.py` | `core` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/core/retrieval.py` | `core` | `main_chain`（核心主入口可达） | core.py | 2（`lingshu/core/contradiction.py`, `lingshu/core/core.py`） | 0 |
| `lingshu/core/structure_permissions.py` | `core` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/core/structure_service.py`） | 0 |
| `lingshu/core/structure_service.py` | `core` | `standalone_leaf`（独立原型叶模块） | — | 0 | 3 |
| `lingshu/core/structure_writer.py` | `core` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/core/structure_service.py`） | 1 |
| `lingshu/core/time_core.py` | `core` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 0 |
| `lingshu/core/tms.py` | `core` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 1 |
| `lingshu/core/verdict.py` | `core` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/gen/hex_composite.py` | `gen` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/gen/hexgen_self_source.py`） | 0 |
| `lingshu/gen/hexgen_align_probe.py` | `gen` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/gen/hexgen_pack_certify.py`） | 2 |
| `lingshu/gen/hexgen_c1_real.py` | `gen` | `subgraph_only`（包内子图互调） | — | 5（`lingshu/gen/hexgen_align_probe.py`, `lingshu/gen/hexgen_multi_seed.py`, `lingshu/gen/hexgen_pack_certify.py`, `lingshu/gen/hexgen_self_source.py` 等 5 处） | 0 |
| `lingshu/gen/hexgen_multi_seed.py` | `gen` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/gen/hexgen_self_source.py`） | 1 |
| `lingshu/gen/hexgen_pack_certify.py` | `gen` | `standalone_leaf`（独立原型叶模块） | — | 0 | 3 |
| `lingshu/gen/hexgen_self_source.py` | `gen` | `subgraph_only`（包内子图互调） | — | 3（`lingshu/gen/hexgen_align_probe.py`, `lingshu/gen/hexgen_pack_certify.py`, `lingshu/gen/hexgen_subject_split.py`） | 3 |
| `lingshu/gen/hexgen_subject_split.py` | `gen` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/nn/hex_cnn.py` | `nn` | `subgraph_only`（包内子图互调） | — | 4（`lingshu/nn/hex_hier.py`, `lingshu/nn/hex_ortho.py`, `lingshu/nn/hex_train.py`, `lingshu/nn/hex_unit.py`） | 0 |
| `lingshu/nn/hex_composite.py` | `nn` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/nn/hex_gen.py`） | 0 |
| `lingshu/nn/hex_gen.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/nn/hex_hier.py` | `nn` | `subgraph_only`（包内子图互调） | — | 4（`lingshu/nn/hex_ortho.py`, `lingshu/nn/hex_query.py`, `lingshu/nn/hex_search.py`, `lingshu/nn/hex_text.py`） | 2 |
| `lingshu/nn/hex_nn_preview.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/nn/hex_ortho.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 3 |
| `lingshu/nn/hex_query.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/nn/hex_recon.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/nn/hex_search.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/nn/hex_text.py` | `nn` | `subgraph_only`（包内子图互调） | — | 2（`lingshu/nn/hex_gen.py`, `lingshu/nn/hex_search.py`） | 1 |
| `lingshu/nn/hex_train.py` | `nn` | `subgraph_only`（包内子图互调） | — | 3（`lingshu/nn/hex_hier.py`, `lingshu/nn/hex_nn_preview.py`, `lingshu/nn/hex_ortho.py`） | 1 |
| `lingshu/nn/hex_unit.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/nn/rust_bridge.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/nn/stcnn.py` | `nn` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/vision/audio.py` | `vision` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 0 |
| `lingshu/vision/image_stats.py` | `vision` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/vision/lineart.py` | `vision` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/vision/video.py` | `vision` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 0 |
| `lingshu/world/anchor_verify.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/semantic_anchor_graph.py`） | 0 |
| `lingshu/world/anchored_verification.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/world/anim.py` | `world` | `main_chain`（核心主入口可达） | WorldModel | 1（`lingshu/world/skeleton3d.py`） | 0 |
| `lingshu/world/autoframe.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 2（`lingshu/world/mesh_svg.py`, `lingshu/world/world3d.py`） | 3 |
| `lingshu/world/brain_store.py` | `world` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/world/scenario_cli.py`） | 1 |
| `lingshu/world/camera_bridge.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/world/camera_calib.py` | `world` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/world/camera_bridge.py`） | 0 |
| `lingshu/world/camera_calibrate.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/world/channel_credibility.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/world/confirmation.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/world/coord_contract.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 6（`lingshu/world/response_probe.py`, `lingshu/world/scene_simulator.py`, `lingshu/world/spacetime_consistency.py`, `lingshu/world/voxel_world.py` 等 6 处） | 0 |
| `lingshu/world/curiosity_explorer.py` | `world` | `main_chain`（核心主入口可达） | core.py | 2（`lingshu/core/core.py`, `lingshu/world/seven_layer_loop.py`） | 1 |
| `lingshu/world/depth_samples.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/world/gap_dual.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 1 |
| `lingshu/world/identity.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 2（`lingshu/world/wm_simloop.py`, `lingshu/world/world_model.py`） | 0 |
| `lingshu/world/mesh3d.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 4（`lingshu/world/autoframe.py`, `lingshu/world/mesh_gen.py`, `lingshu/world/mesh_svg.py`, `lingshu/world/world3d.py`） | 1 |
| `lingshu/world/mesh_gen.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/shapes.py`） | 3 |
| `lingshu/world/mesh_svg.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 3 |
| `lingshu/world/motion.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 3（`lingshu/world/physics_predict.py`, `lingshu/world/world_learner.py`, `lingshu/world/world_model.py`） | 0 |
| `lingshu/world/motion_groups.py` | `world` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/world/__init__.py`） | 0 |
| `lingshu/world/multiview.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/world3d.py`） | 0 |
| `lingshu/world/physics.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 4（`lingshu/world/camera_calibrate.py`, `lingshu/world/mesh_gen.py`, `lingshu/world/physics_predict.py`, `lingshu/world/scene_simulator.py`） | 0 |
| `lingshu/world/physics_predict.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/world_model.py`） | 3 |
| `lingshu/world/prediction.py` | `world` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 0 |
| `lingshu/world/process_ops.py` | `world` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/world/__init__.py`） | 1 |
| `lingshu/world/response_probe.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/world/scenario.py` | `world` | `subgraph_only`（包内子图互调） | — | 1（`lingshu/world/scenario_cli.py`） | 1 |
| `lingshu/world/scenario_cli.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 2 |
| `lingshu/world/scene_model.py` | `world` | `main_chain`（核心主入口可达） | WorldModel | 2（`lingshu/world/brain_store.py`, `lingshu/world/scenario.py`） | 3 |
| `lingshu/world/scene_simulator.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 6（`lingshu/core/core.py`, `lingshu/world/seven_layer_loop.py`, `lingshu/world/spacetime_consistency.py`, `lingshu/world/wm_simloop.py` 等 6 处） | 3 |
| `lingshu/world/semantic_anchor_graph.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/world3d.py`） | 1 |
| `lingshu/world/seven_layer_loop.py` | `world` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 3 |
| `lingshu/world/shade_paint.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 1（`lingshu/world/mesh3d.py`） | 0 |
| `lingshu/world/shapes.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 3（`lingshu/world/autoframe.py`, `lingshu/world/mesh_gen.py`, `lingshu/world/world3d.py`） | 1 |
| `lingshu/world/silhouette3d.py` | `world` | `main_chain`（核心主入口可达） | WorldModel | 1（`lingshu/world/scene_model.py`） | 3 |
| `lingshu/world/skeleton3d.py` | `world` | `main_chain`（核心主入口可达） | WorldModel | 2（`lingshu/world/scene_model.py`, `lingshu/world/silhouette3d.py`） | 2 |
| `lingshu/world/spacetime_consistency.py` | `world` | `main_chain`（核心主入口可达） | core.py | 2（`lingshu/core/core.py`, `lingshu/world/world_learner.py`） | 2 |
| `lingshu/world/stable_lease.py` | `world` | `standalone_leaf`（独立原型叶模块） | — | 0 | 0 |
| `lingshu/world/svg_backend.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 3（`lingshu/world/mesh_svg.py`, `lingshu/world/silhouette3d.py`, `lingshu/world/world3d.py`） | 0 |
| `lingshu/world/time_core.py` | `world` | `main_chain`（核心主入口可达） | core.py | 2（`lingshu/world/gap_dual.py`, `lingshu/world/seven_layer_loop.py`） | 0 |
| `lingshu/world/voxel_world.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 2（`lingshu/core/core.py`, `lingshu/world/scene_simulator.py`） | 1 |
| `lingshu/world/vprim.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 2（`lingshu/core/core.py`, `lingshu/world/world3d.py`） | 0 |
| `lingshu/world/wm_simloop.py` | `world` | `main_chain`（核心主入口可达） | core.py | 1（`lingshu/core/core.py`） | 3 |
| `lingshu/world/world3d.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 7（`lingshu/core/core.py`, `lingshu/world/autoframe.py`, `lingshu/world/camera_bridge.py`, `lingshu/world/depth_samples.py` 等 7 处） | 7 |
| `lingshu/world/world_learner.py` | `world` | `main_chain`（核心主入口可达） | core.py | 3（`lingshu/core/core.py`, `lingshu/world/curiosity_explorer.py`, `lingshu/world/response_probe.py`） | 4 |
| `lingshu/world/world_model.py` | `world` | `main_chain`（核心主入口可达） | core.py, WorldModel | 3（`lingshu/core/core.py`, `lingshu/world/physics_predict.py`, `lingshu/world/wm_simloop.py`） | 5 |
| `lingshu/world/world_process.py` | `world` | `subgraph_only`（包内子图互调） | — | 2（`lingshu/world/__init__.py`, `lingshu/world/process_ops.py`） | 0 |

## 四、包初始化模块（6 个 `__init__.py`）

| 路径 | 层 | 包内入度 | 包内出度（重导出/注册目标） |
|---|---|---:|---|
| `lingshu/__init__.py` | `root` | 0 | `lingshu/_pathguard.py` |
| `lingshu/core/__init__.py` | `core` | 0 | — |
| `lingshu/gen/__init__.py` | `gen` | 0 | — |
| `lingshu/nn/__init__.py` | `nn` | 0 | — |
| `lingshu/vision/__init__.py` | `vision` | 0 | — |
| `lingshu/world/__init__.py` | `world` | 0 | `lingshu/world/motion_groups.py`, `lingshu/world/process_ops.py`, `lingshu/world/world_process.py` |

## 五、诚实边界说明

1. 本台账按当前仓库树上 AST 静态/惰性 import 实测生成：凡未从核心主入口导入的模块均如实标为 `subgraph_only` 或 `standalone_leaf`，不把「有单测覆盖」伪装为「已接入主链」。
2. `lingshu/world/__init__.py::register_world_process(engine)` 采用依赖倒置（调用方显式把 `world_process` / `motion_groups` / `process_ops` 注册给 `SpacetimeMemoryEngine` 实例），静态图中 `core.py` 不反向导入它，故三模块在此客观记为 `subgraph_only`（由 `lingshu/world/__init__.py` 导入）。
3. `lingshu/vision/__init__.py` 通过相对动态导入延迟暴露 `lineart` 符号；默认 AST 扫描仅解析静态导入及字面绝对 `lingshu.*` 动态导入，因此该边界未计入包内图，`lineart.py` 按当前扫描结果保持 `standalone_leaf`。默认 `--check` 仅锁定 `lingshu/` 包内拓扑；查看 tests/bench 外部引用可运行 `python -X utf8 tools/module_topology_audit.py --markdown --with-test-refs` 或 `--json`。
4. 若后续 PR 修改 `lingshu/` 模块集合或包内跨模块导入关系（如切断 `world -> core` 反向边或将原型模块接入主链），须同步执行 `python -X utf8 tools/module_topology_audit.py --write` 更新本台账，否则 `--check` 门禁将精确报出漂移；仅增删 `tests/` 或 `bench/` 文件不会触发本门禁漂移。
