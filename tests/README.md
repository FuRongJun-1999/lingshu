# tests/ · 运行方式与已知状态

> 本文档登记测试件的双模式运行方式、已知红与耗时档位。改动结论均以**能被独立复跑**为准。

## 一键运行

```bash
pip install -e ".[full,dev]"     # world/nn/gen 需 numpy+Pillow；dev 提供 pytest
python -m pytest tests/ -v -rs --ignore=tests/test_hex_search.py --ignore=tests/test_hex_hier.py
                                 # ↑ 快档（≈80s）；两个重件见下表
python -m pytest tests/ -v -rs   # 全量（≈20min，含两个重件）
```

## 双模式说明

测试件支持两种入口，断言集一致：

| 形态 | 文件 | 说明 |
|---|---|---|
| pytest 原生 | `test_hex_ortho` `test_hex_search` `test_hex_text` `test_hex_train` `test_hex_hier` | `python -m pytest <file> -v`；直接 `python <file>` 亦转投 pytest |
| 脚本/pytest 双模式 | `test_hex_cnn` `test_hex_composite` `test_hex_gen` `test_brain_store` `test_wm_verify_unobserved` | `python tests/test_<x>.py` 原样保留（intake 复跑命令不变）；执行体收拢在 `main()`/`run_checks()`，pytest 经薄包装入口收集（见下"改造注记"） |

前置说明：`test_brain_store` 需环境变量 `MDCG_BRAIN_PYTHONPATH`（脑包目录）才全跑，
缺前置时显式 SKIP 并以 0 退出——CI 中该件恒为 SKIP，不阻塞。

## 耗时档位（上游 intake v0.2 §四实测；本地 py3.12/numpy2.5.1 同量级）

| 档位 | 文件 | 耗时 |
|---|---|---|
| 快档 | 其余全部 | 合计 ≈80s |
| 重件 | `test_hex_search` | ≈361.6s |
| 重件 | `test_hex_hier` | ≈657.5s |

CI（`.github/workflows/gate.yml`）按此跑 pytest 腿：py3.12（阻断）＋ py3.13 与
「依赖下界」两档 advisory，串行全量收集（排除下方脚本式件由 gate 自身排除表管理）；
PR 另有 `pr-lint.yml` 标题门与 `repro-bot.yml` 复跑机器人。本地全量可用
`python -m pytest tests/ -n auto`（dev extra 的 pytest-xdist，实测 483s → 228s；
时序敏感件在满载下的口径见各件 docstring）。

## 曾登记的「已知红」——已转绿（2026-10-11 更新）

`test_hex_text` 曾登记 2 项已知红：`test_spatial_detect_finds_single_object`（6/16）、
`test_multimodal_check_clean_vs_corrupt`（7/12）——**根因修复已落 main**：
`9bc4985`（`image_to_grid` 晶格横向越出画幅 √3 倍，采纳自 PR #116）＋ `54cb5cb`
（`hex_hier` 形状辅助损失按 OBJ 布局连续分组）。两条现随 `tests/` 自动收集真跑：
本机 `python -X utf8 -m pytest tests/test_hex_text.py -q` = **6 passed**。
`gate.yml` 上曾挂的 `--deselect` 已摘除；守卫 `tests/test_gate_deselect_justified.py`
断言「凡被 deselect 的用例必须真红」——再 deselect 一条绿的用例会当场判红。
原读数与讨论见 issue #4 / #24（判据「不可复算」的批评）。

## 改造注记（2026-10-08 · 测试基建）

- `test_hex_cnn` 原为模块级直跑脚本——pytest 收集时即触发 `sys.exit`，
  **整个 `tests/` 目录无法一键收集**（INTERNALERROR）。现执行体收拢进 `main()`，
  脚本模式退出码语义不变；本地读数 `13 passed, 0 failed`（改造前后逐位一致）。
- `test_brain_store` / `test_wm_verify_unobserved` 原有 `main()` 但 pytest 收不到
  用例；各加一个薄包装 `test_*()`（`assert main() == 0`），断言零改动。
- `test_hex_text` 两项已知红曾提议 xfail 标注——PR #87 部分采纳时**明确未取**
  （xfail 从未落仓，根因由 `9bc4985`/`54cb5cb` 真修取代，见上节）；
  全仓测试件至今**零 xfail 标注**（守卫 `tests/test_issue446_readme_reality.py`
  钉「文档声称转绿 ⇔ 代码确无 xfail」两侧一致）。
- 新增根 `conftest.py`（sys.path 注入）已进 main；同 PR 自带的 ci.yml 工作流
  **未被采纳**（与 main 的 gate.yml 重复且更窄，该文件从未落仓）——CI 真源为
  `gate.yml`。
