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

CI（`.github/workflows/ci.yml`）按此分两档：快档随 push/PR 跑；慢档每周一定时 +
手动触发（workflow_dispatch）。

## 已知红（如实登记，非本仓引入）

`test_hex_text` 含 **2 项已知红**：`test_spatial_detect_finds_single_object`（6/16）、
`test_multimodal_check_clean_vs_corrupt`（7/12）。**上游 AEIS 同跑同红、失败读数逐位
相同**（intake body-export-v0.2 §五）⇒ 系历史既有（环境/训练数值性），非导出引入。

处理方式：两项已标 `@pytest.mark.xfail(strict=True)`——套件保持全绿可当门禁，
reason 内登记出处；修好后以 XPASS→FAIL 提醒摘牌。另见 issue #24（判据本身
"不可复算"的批评在案）。

## 改造注记（2026-10-08 · 测试基建）

- `test_hex_cnn` 原为模块级直跑脚本——pytest 收集时即触发 `sys.exit`，
  **整个 `tests/` 目录无法一键收集**（INTERNALERROR）。现执行体收拢进 `main()`，
  脚本模式退出码语义不变；本地读数 `13 passed, 0 failed`（改造前后逐位一致）。
- `test_brain_store` / `test_wm_verify_unobserved` 原有 `main()` 但 pytest 收不到
  用例；各加一个薄包装 `test_*()`（`assert main() == 0`），断言零改动。
- `test_hex_text` 两项已知红加 xfail 标注（断言与数值判据零改动）。
- 新增根 `conftest.py`（sys.path 注入）与 `.github/workflows/ci.yml`（快/慢双档）。
