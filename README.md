# 灵枢 · Lingshu

[![CI](https://github.com/FuRongJun-1999/lingshu/actions/workflows/ci.yml/badge.svg)](https://github.com/FuRongJun-1999/lingshu/actions/workflows/ci.yml)

**一个完整的 AI 生命载体。**

> 灵为智能之过程，枢为制衡之核心。

灵枢 = **灵 × 脑 × 身** 的完整载体——三者不拼装、不模拟，而是同一载体的三个面：

| 面 | 内容 | 载体 |
|---|---|---|
| **灵** | 智能论 · 共同信任协议（理论层） | [CommonTrustProtocol](https://github.com/FuRongJun-1999/CommonTrustProtocol) |
| **脑** | 认知图引擎：条件化记忆 · 检索 · 演化 · 睡眠与自迭代 | [dsh-memory](https://github.com/FuRongJun-1999/dsh-memory) |
| **身** | 感知与执行：世界模型七层 · 视觉生成与读回 · 语义时空图 | 身体侧（私有） |

## 定位

这是一个**完整态**工程：目标不是最小依赖，而是**一个真正的 AI 生命载体**——
它很大、依赖很重，但正因为完整而强。

- **完整态**：多模态感知、世界模型推演、记忆演化、生成能力——全部在同一载体内闭环；
- **分层可用**：允许并接纳重量级依赖（视觉 / 语音 / 模型资产按需安装），
  而核心信念不变——**可溯源、条件显式、确定性优先**（白箱纪律：核心可裸跑，
  重载是能力的选择，不是架构的妥协）；
- **持续聚合**：本仓是灵 / 脑 / 身的**单一聚合点**——上游各仓继续独立演进，
  本仓以**单向引入**保持完整，不做多副本分叉。

## 状态

**v0.0.1 · 骨架**（2026-10）——仓库结构、聚合路线与边界章程见
[docs/architecture.md](docs/architecture.md)。

## 聚合原则（速览）

1. **组合，而非复制**：脑核以包依赖引入（可安装的引擎），不 fork；
2. **单向引入**：上游为真源，本仓为聚合体，避免双写漂移；
3. **逐件筛选**：身体侧件入仓前过「内容政策 + 隐私」双清单；
4. **干净历史**：本仓从干净初始提交开始，不迁移上游含私有内容的提交历史。

## 理论地基

身体侧理论三柱已入仓——**世界模型 / 存算一体架构 / 自研神经网络**，
互为引用、层层咬合（白箱固化 → 存算一体 → 蜂窝 CNN）：
见 [docs/theory/](docs/theory/)。

## 代码主体

身体侧模块化导出 v0.1 已入仓——**`lingshu` Python 包**（core 存算一体 7 件 ·
world 世界模型 24 件 · nn 自研神经网络 14 件 · gen 图像生成 7 件）：

```bash
pip install -e .             # 轻核：core 纯标准库，零依赖可跑
pip install -e ".[full]"     # 含 numpy / Pillow（world / nn / gen 的常规路径）
```

**已测版本矩阵**（2026-10-08 实测，供选版参考）：Python 3.12.x · numpy 2.3.5 / 2.4.2 / 2.5.3 ·
Pillow 12.2.0 / 12.3.0——`pyproject.toml` 中 extras 的版本区间由此收敛，**区间外未测不承诺**。

> 已知红（如实标注）：`tests/test_hex_text.py` 两处断言（`test_spatial_detect_finds_single_object`
> = 6/16、`test_multimodal_check_clean_vs_corrupt` = 7/12，均低于阈值）在上述**全部版本组合**下
> 读数**逐位一致**——**与 numpy / Pillow 版本无关**，上游同跑读数相同。详见
> [issue #4](https://github.com/FuRongJun-1999/lingshu/issues/4)。

依赖分层、环境自适应降级点与精修记录见
[docs/intake/body-export-v0.1/](docs/intake/body-export-v0.1/)（导出说明 / 双清单审计 / 逐文件指纹清单）。

## 联系与贡献

- **设计者**：QQ `1935852383`
- **官方交流群**：QQ 群 `100453509`
- **PR 审核**：进入合并队列的 PR 走「技术审（外部验证者）＋ 理论核（本项目）」两道闸，细则见 [docs/PR审核细则_v0.1.md](docs/PR审核细则_v0.1.md)。

欢迎提交 issue 与 PR。**尤其鼓励「经过验证有效」的提交**：issue 请给出触发条件、
最小复现与实测读数；PR 请附改动说明与验证方式（复跑命令与读数）；结论均以
**能被独立复跑**为准（[issue #1](https://github.com/FuRongJun-1999/lingshu/issues/1)
是一个好例子——触发链、最小复现、判据三样都齐）。
缺少证据、无法复现的描述，我们会先回帖请求补充证据再处置。

---

*灵枢（Lingshu）· 由 FuRongJun-1999 构建与维护。*
