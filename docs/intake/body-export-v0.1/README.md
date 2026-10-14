# lingshu 导出物 v0.1 · 四块（世界模型 / 存算一体 / 自研神经网络 / 图像生成）

> 生成：2026-10-02 · 来源：身体侧真源（AEIS）· 生成工具：`AEIS/tools/public_export/`
> 形态：**单向引入材料**（对接 lingshu 聚合仓章程）——上游保持私有主体，本导出为筛选后的固化副本。
> **双清单审计：CLEAN（0 命中）** —— 见 `AUDIT.md`（含复核先例）。
> **冒烟：51/51 模块 import 通过 + 构造性运行通过**（`python smoke.py` 可复跑）。

---

## 一、内容与结构

```
lingshu/
├── core/   存算一体（7 件）：协议实例核心引擎 core / 时间核 time_core / 激活层 activation /
│           长期门 longterm_gate / 裁决 verdict / 条件归一化 condition_normalize / 溯源 provenance
├── world/  世界模型（24 件）：推演循环 wm_simloop / 统一世界模型 world_model / V-JEPA 学习器 /
│           好奇探索 / 七层闭环 / 4D 沙盒 voxel_world / 场景模拟 / 时空一致 / 预测 / 锚定验证（弱强+n感知机）/
│           语义锚点图 / WORLD3D / 多视角 / 视觉原语 vprim / 形状库 / D_task-D_meta / 骨架-剪影 3D
├── nn/     自研神经网络（14 件）：HEX-CNN 全家族（cnn / hier / train / search / ortho / composite /
│           text / gen / unit / preview / query / recon）+ 3D 时空卷积 stcnn + Rust 桥 rust_bridge
└── gen/    图像生成（7 件）：确定性渲染器 v3 hexgen_self_source / 主体分割 hexgen_subject_split /
            读出器与语料解析 hexgen_c1_real / 对齐探针 hexgen_align_probe / 铺位穷举证书 hexgen_pack_certify /
            多 seed 评测 hexgen_multi_seed
```

## 二、依赖与重载声明（对接章程「轻核 / 重载」分层）

| 层 | 重依赖 | 可选重载 |
|---|---|---|
| core | **无（纯标准库）** ✅ 轻核承诺成立 | `numpy`+`scipy.sparse`（activation 激活层；缺失时降级 edge-list 传播，源代码内已声明局限） |
| world | `numpy` | `PIL`（渲染相关子功能） |
| nn | `numpy`、`PIL` | Rust 后端可执行文件（rust_bridge；缺失时纯 Python 路径） |
| gen | `numpy`、`PIL` | `torch`（**仅** hexgen_multi_seed 的语料生成函数内，测试/评测用；渲染与读回不依赖） |

## 三、环境自适应降级点（如实登记，非缺陷）

- **core 保留 5 处「未随本导出」的扩展接口**（`vision` / `game_web.generate` / `game_web.semantics`）：
  运行期检测，缺失时返回 `BLINDSPOT` / `gen_not_ready`——这是上游**原生的公开形态设计**
  （源码注释原文：「没有则诚实声明 BLINDSPOT（公开仓库形态）」）。
- **core 保留 10 个「未实现扩展点」名义**（`flywheel_engine` / `prediction_engine` /
  `self_cognition_engine` / `lifecycle_engine` / `attention_policy` / `entity_registry` /
  `cognitive_orchestrator` / `blindspot_learning_loop` / `scene_reconstruction` / `semantic_space`）：
  全部在函数内 `try/except` 降级，未装入即为空实现。
- **导出精修（一次性，逐条记录于 `_export_manifest.json`）**：
  ① `core.py` 内联 `_bigram_set` 三行工具函数（原借自私有侧 LLM 路由模块——白箱化）；
  ② `activation.py` 数据路径由本机字面量 → 环境变量 `LINGSHU_DATA_ROOT`（缺省 `data/`）；
  ③ `provenance.py` 移除私有侧数据源分支（会话域前缀/观测工具/来源映射三处 + 枚举表两处）。

## 四、导入约定

- 整包：`from lingshu.world import world_model`；`import lingshu.gen.hexgen_self_source`
- 引用形式已导出器改写：**块内** `from .x import ...`；**跨块** `from ..core.time_core import ...`
- 按层单装：**core 自含**（可单独安装、单独运行）；world/nn/gen 若拆分需要 core 层随带
  （源码内 `try 同块 / except 跨层` 双保险已就位，如 `gap_dual.py` 的 `time_core`）。

## 五、数据与测试（如实边界）

- **数据**：`gen` 中「语料依赖函数」（黑箱图评测 `hexgen_c1_real` 的评测口、`hexgen_multi_seed` 的多 seed
  指标）需要外部语料目录（上游 `data/vision/*`），**未随导出**。构造性能力自带零数据：
  渲染 `render_prompt_v3` / `shape_mask_v3`、分割 `split_subjects`、证书 `hexgen_pack_certify`。
- **测试**：上游守门测试（hexgen 20+ 套、世界模型 27 套、六块合计 90+ 套）**未随本次导出**——
  下批按「构造性 vs 语料依赖」分拣后迁移。本次验证＝import 冒烟 51/51 ＋ 构造性运行（渲染圆→分割读回）。

## 六、复核记录

- 双清单机械审计：`AUDIT.md`（0 命中）；误报复核先例（`SelfModel`/`_self_cognition`/`f://enum`）记录在
  审计器源码注释中。
- 导出清单与逐文件指纹：`_export_manifest.json`（含每文件的改写数、精修数、隐藏引用扫描结果）。
