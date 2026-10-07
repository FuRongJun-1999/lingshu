# lingshu 导出物 v0.2 · 固化物 + 测试件（构造性）

> 生成：2026-10-08 · 来源：身体侧真源（AEIS）· 生成工具：`AEIS/tools/public_export/`
> （沿用 v0.1 做法；本批最小增量扩展见 §三）· 形态：**单向引入材料**（对接聚合仓章程）——
> 上游保持私有主体，本导出为筛选后的固化副本；**不改写 v0.1 历史件**（单向引入）。
> **双清单审计：新增面 CLEAN（0 命中）**——见 `AUDIT.md`（附既有面复扫发现栏）。
> **冒烟与实测**：读数见 §四（逐件实跑，含已知边界）。

---

## 一、内容

| 面 | 件 | 说明 |
|---|---|---|
| 固化物 | `lingshu/nn/assets/kernels_img0.json`（1 件） | 理论稿《自研蜂窝CNN_理论稿_v0.1》§一.2/§四点名的白箱固化产物（「JSON 可直读，加载即确定性模块」）。**字段级脱敏 1 处**：`source` 素材编号 → `training-image 240x394`（数值/核参数未动）。 |
| 测试件 | `tests/`（8 件）：`test_hex_cnn` / `_composite` / `_gen` / `_hier` / `_ortho` / `_search` / `_text` / `_train` | 理论稿点名证据（cnn 25/25、ortho 11/11 等）的**构造性**部分——v0.1 声明「下批按『构造性 vs 语料依赖』分拣后迁移」的执行。 |
| 元数据 | `pyproject.toml` | 增 `package-data`（`lingshu.nn` → `assets/*.json`）与 `dev` extra（`pytest`）。 |

**不随本批**（判据：语料依赖或未导出件依赖，宁可少导不可漏筛）：

- `test_hexgen_align_probe` 家族 12 件（r298–r306、r311、r312）：判据经 `load_items(CORPUS)` 读黑箱语料（`data/vision/qwen_controlled`）；r311/r312 另依赖 `PYTHONHASHSEED` 跨进程复现。
- `test_hexgen_subject_split` r318–r321 4 件：`glob data/vision/qwen_multi/*/*.png` 私有数据集路径硬编码。
- `test_hexgen_c1_real` / `_multi_seed` 2 件：读语料库文件 `connection_library.json`。
- `test_hexgen_pack_certify_r313` 1 件：断言实验证书落盘文件在盘。
- `test_hexgen_p1_detect` 1 件：硬依赖未导出件（`hexgen_p1_detect` / `hexgen_p1_readback` / `query_precision`）。
- `test_hexgen_self_source` 1 件（边界件）：11 项中 10 项构造性、1 项（闭环）依赖黑箱标定语料——本批排除；skip 化方案已试点通过（10 OK + 1 显式 SKIP），列为下批候选。

## 二、模块刷新判定（本批实测：无过时，不刷新）

- 方法：重跑上游 `export.py` 管线（world/core/nn/gen 四块）输出与仓内**逐件二进制比对 + 逐行 diff**。
- 读数：**52 件中 50 件字节级一致（SAME）**；差异 2 件均为本仓既有 `f8b69ab`「公开前脱敏处置」改写
  （`nn/hex_gen.py` 内部工作流号→「内部调研」；`world/silhouette3d.py` 措辞中性化）——
  **若按上游源刷新这 2 件将回归脱敏破坏**，故不刷新。另 `world/brain_store.py` 为本仓新件（`fb4b98b`），无上游对照。
- 记录：「AEIS 侧字节数更大 ⇒ 上游过时」经复核**否证**——字节差 = import 改写长度差
  （如 `from aeis.hex_train import` → 相对导入，短若干字节），非上游演进；重管线产物与仓内件逐件一致。

## 三、导出改写与精修（逐条，可复跑复算）

1. **测试件 import 改写**：`from aeis.hex_X import` → `from lingshu.nn.hex_X import`（跨层约定同 v0.1）。
2. **仓根引导块**：5 件（hier/ortho/search/text/train）补 `sys.path` 引导（原以 `experiments`/`.` 相对注入，导出面不成立）；
   3 件（cnn/composite/gen）原有根路径注入，保留。
3. **`test_hex_cnn.py` 脱敏 patch（11 处改写）**：
   - 段④⑤⑥（11 项，依赖上游素材 `data/img/0.png`）整体包 `if os.path.exists(_ASSET)` / `else: 显式 SKIP 打印`——
     **素材在盘（上游）时原判据逐字全跑；导出面无素材时诚实 SKIP，不伪造通过**（对齐仓内「环境自适应降级点」先例）；
   - 旧机器路径注释、`nahida_work` 叙事、「人物/立绘/纳西妲」措辞、`source` 字段值、变量名 `img0` → 中性化。
4. **sanitize 规则**：沿用 v0.1 清扫表；**新增 1 条**：「Program Files 下 2_ai 工作区」形态的旧机器**正斜杠**路径（原兜底正则只覆盖反斜杠形态；本规则触发于 `test_hex_cnn.py` 的旧注释，patch 后该注释本身已中性化）。
5. **工具扩展（过程物，不改上游工具）**：`audit_intake_ext.py`——`.json` 纳入扫描面（v0.1 跳过 `.json` 会漏固化物）+
   `.git` 内部排除 + 报告输出路径参数；**PRIVATE 名单与 PATS 与原版逐字一致（不放宽）**。
6. 逐件指纹/改写数/脱敏数：`_export_manifest.json`（本批 8+1 件逐件 sha256 与计数）。

## 四、冒烟与实测（正式面逐件实跑）

| 项 | 读数（实跑） | 命令 |
|---|---|---|
| 模块 import 冒烟 | **ok=51 fail=0** + 构造性运行（圆掩膜 8137 px；分割读回 1 件） | `python <AEIS>/tools/public_export/smoke.py <本仓根>` |
| 固化物加载 | 6 核全加载（各 7 参数）；**导出副本与源 `allclose(atol=0) = True`**、键集一致（脱敏未动数值） | 见 §六 |
| `tests/test_hex_cnn.py` | **13 passed, 0 failed** + 段④⑤⑥ 显式 SKIP（素材在盘时全 25 项） | `python tests/test_hex_cnn.py` |
| `tests/test_hex_composite.py` | **24/24 通过** | `python tests/test_hex_composite.py` |
| `tests/test_hex_gen.py` | **30/30 通过** | `python tests/test_hex_gen.py` |
| `tests/test_hex_hier.py` | **7 passed**（657.5s） | `python tests/test_hex_hier.py`（需 pytest） |
| `tests/test_hex_ortho.py` | **11 passed**（0.11s）——与理论稿「11/11」一致 | 同上 |
| `tests/test_hex_search.py` | **7 passed**（361.6s）——理论稿 §八.4「18 passed」复现（11+7） | 同上 |
| `tests/test_hex_text.py` | **2 failed / 4 passed**——已知红（上游同红、读数逐位相同，见 §五） | 同上 |
| `tests/test_hex_train.py` | **9 passed**（5.5s） | 同上 |

## 五、如实边界

- `test_hex_text.py` **含 2 项已知红**（`test_spatial_detect_finds_single_object`、
  `test_multimodal_check_clean_vs_corrupt`）：**上游 AEIS 同跑同红、失败读数逐位相同**（6/16、7/12 及 verdicts 序列一致）
  ⇒ 系历史既有（环境/训练数值性），**非导出引入**；导入以本注记为准，不视为导出面缺陷。
- `test_hex_cnn.py` 导出面读数 = 13 项构造性 + 段④⑤⑥ 显式 SKIP（合计 25 项结构与理论稿一致；素材在盘时全跑）。
- 固化物为脱敏副本：`source` 字段值与源文不同（1 处），**核数值逐位相同**（加载实测见 §四）。
- 本批不含模块刷新（§二：无过时）；不含语料依赖测试件（§一：下批候选）。

## 六、复跑

```bash
# 模块 import 冒烟 + 构造性运行（工具：AEIS/tools/public_export/smoke.py）
python <AEIS>/tools/public_export/smoke.py <本仓根>
# 固化物加载（确定性模块）
python -c "import sys; sys.path.insert(0,'.'); from lingshu.nn.hex_cnn import load_consolidated; \
print(sorted(load_consolidated('lingshu/nn/assets/kernels_img0.json')))"
# 测试件（需 pytest：pyproject dev extra）
python -m pytest tests/test_hex_ortho.py tests/test_hex_search.py -v   # 理论稿 §八.4 点名
python tests/test_hex_cnn.py      # 13 项 + 素材段 SKIP
python tests/test_hex_composite.py && python tests/test_hex_gen.py
```
