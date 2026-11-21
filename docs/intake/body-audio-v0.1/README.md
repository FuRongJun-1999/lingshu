# 身体音频感知 v0.5 · 活动周期增量

本次在既有 PR #1460 上追加 activity_period 纯函数、可观测前沿约束与降采样保守拒绝；之前记录的 WAV/FFmpeg 能力也在同一 PR 内。

来源：本仓新实现，2026-11-19；非上游导出件。身体侧 WAVE PCM 声学摘要模块。

新增件：
- lingshu/vision/audio.py：标准库 WAV 分析、内存 bytes/bytearray/memoryview 输入和摘要写入助手；
- SpacetimeMemoryEngine.perceive_audio：对接既有 audio 模态感知写入；
- segment_audio_events：将有界 RMS 包络确定性地切成 activity/silence 片段，并显式标注降采样时间精度；
- infer_audio_activity_period：基于 RMS 的可观测静音→活动前沿估计重复周期；证据/分辨率不足返回空周期与 reason，不将结果解释为语音韵律、音乐节拍或世界事件；
- analyze_audio / perceive_audio_file：除 WAV 标准库路径外，可按需用 FFmpeg 支持本地 MP3/M4A/AAC/OGG/Opus/FLAC/AIFF/WMA/WebM 解码；
- tests/test_world_audio_perception.py：合成 WAV 构造性与边界验收；
- docs/world_audio_perception.md：公开用法、范围与复跑说明。

## 内容清单

- WAV 分析与事件画像仅使用 Python 标准库；压缩格式调用可选宿主 FFmpeg 子进程；不增加 Python 音频包、模型资产或网络服务依赖。
- 测试 WAV 在 pytest 临时目录里由测试代码合成；仓库中不包含录音或真实音频素材。
- 特征提取保留声学统计与有界能量包络，派生活动/静音分段和保守周期估计；周期仅是阈值上升沿间隔，不是语音韵律、音乐节拍、语义事件或世界模型推断。
- API 明确限制 WAV PCM 位深/声道/时长/输入大小；可选 FFmpeg 解码另有限定 demuxer、pipe-only 协议、64 MiB 输入/输出、最长 3600 秒和 600 秒超时；无 FFmpeg/畸形输入均 fail-closed。

## 隐私清单

- 不读取或提交用户记忆、会话、录音、路径配置、凭据或身份文件。
- 文件输入按块读取；bytes-like 输入直接从内存缓冲解析，不落临时文件；摘要正文、结构化特征和返回对象均不含输入路径或文件名。
- 引擎只通过既有 add_perception(..., modality="audio") 写入摘要；FFmpeg 仅从本地 bytes pipe 读取，不联网、不落临时文件；不保存原始音频、不上传数据。
- 无第三方受限内容；算法为本仓新写的 PCM 解码与统计量计算。

## 验证面

- 合成 1 秒静音—恒音—静音 WAV：校验声学统计、两段静音区间及 activity/silence 事件画像。
- 覆盖事件分辨率、FFmpeg 安全边界与 MP3 往返；周期估计测试稳定等间隔脉冲、不规则前沿、持续高电平无前沿、少于 4 个前沿、静音、真实长录音降采样拒绝、11,025 Hz 实际窗长换算及记忆摘要闭环。
- 覆盖 8/16/24/32-bit PCM、bytes/bytearray/memoryview 内存输入及无临时文件闭环、立体声满幅样本、损坏/截断输入、时长上限、摘要入 audio 模态及路径不泄漏。
- 标准库导入子进程显式阻断 numpy/Pillow，验证轻核音频入口不触发图像重依赖。
- 复跑：python -X utf8 -m pytest tests/test_world_audio_perception.py -q --no-header（含事件画像、FFmpeg mock 边界及真实 MP3 往返）。
- 身体包烟测：python -X utf8 tools/intake_smoke_min.py --quiet（将包根也计入，合入当前主线后为 91/91（2026-11-21））。

范围不包括 ASR/转写、说话人识别、实时设备采集、网络媒体 URL、原始音频持久化或语义事件分类。
