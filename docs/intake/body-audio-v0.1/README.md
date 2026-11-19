# 身体音频感知 v0.1 · 本仓新件登记

来源：本仓新实现，2026-11-19；非上游导出件。身体侧 WAVE PCM 声学摘要模块。

新增件：
- lingshu/vision/audio.py：标准库 WAV 分析和摘要写入助手；
- SpacetimeMemoryEngine.perceive_audio：对接既有 audio 模态感知写入；
- tests/test_world_audio_perception.py：合成 WAV 构造性与边界验收；
- docs/world_audio_perception.md：公开用法、范围与复跑说明。

## 内容清单

- 仅使用 Python 标准库 wave / struct / math；无模型资产、第三方音频运行依赖或外部服务。
- 测试 WAV 在 pytest 临时目录里由测试代码合成；仓库中不包含录音或真实音频素材。
- 特征提取只保留时长、采样率/位深、RMS/峰值、活动/静音分段和有界能量包络；不做语音识别。
- API 明确限制 PCM 位深/声道/时长/输入大小；畸形和超预算输入 fail-closed，不静默降级。

## 隐私清单

- 不读取或提交用户记忆、会话、录音、路径配置、凭据或身份文件。
- 原始音频只流式读取；摘要正文、结构化特征和返回对象均不含输入路径或文件名。
- 引擎只通过既有 add_perception(..., modality="audio") 写入摘要；不保存音频字节、不上传数据。
- 无第三方受限内容；算法为本仓新写的 PCM 解码与统计量计算。

## 验证面

- 合成 1 秒静音—恒音—静音 WAV：校验声学统计和两段静音区间。
- 覆盖 8/16/24/32-bit PCM、立体声满幅样本、损坏/截断输入、时长上限、摘要入 audio 模态及路径不泄漏。
- 标准库导入子进程显式阻断 numpy/Pillow，验证轻核音频入口不触发图像重依赖。
- 复跑：python -X utf8 -m pytest tests/test_world_audio_perception.py -q --no-header。
- 身体包烟测：python -X utf8 tools/intake_smoke_min.py --quiet（将包根也计入，当前 90/90）。

范围不包括 ASR/转写、说话人识别、压缩音频、流式设备采集或原始音频持久化。
