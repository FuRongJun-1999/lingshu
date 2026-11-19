# 身体音频感知 v0.5 · 活动周期估计

lingshu.vision.audio 提供 PCM WAV 声学分析、可选 FFmpeg 压缩音频适配和保守的声学活动周期估计，属于身体侧感知模块。它不转写内容、不推断说话人，也不把原始音频或输入路径写入灵枢记忆。

## Python 接入

SpacetimeMemoryEngine.perceive_audio 支持 WAV 文件路径或 WAV bytes-like 缓冲，无需临时文件；它也可通过可选 FFmpeg 解码常见压缩文件/bytes。perceive_audio_file(engine, source, ffmpeg="ffmpeg", ...) 显式调用统一入口。WAV 不需 FFmpeg；压缩输入在 FFmpeg 不可用时抛 AudioCodecUnavailable。

    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    result = engine.perceive_audio("sample.wav")
    # Optional local FFmpeg for compressed formats: engine.perceive_audio("sample.mp3")
    assert result["raw_audio_stored"] is False

结果包含时长、采样率、声道、位深、RMS/峰值 dBFS、活动比例、削波比例、有界能量包络、静音区间、事件画像、activity_period 估计及摘要节点 id。摘要以 audio 模态、audio_summary 标签写入既有感知入口。

segment_audio_events(features, silence_dbfs=-45) 可独立把能量包络压成按时间排序的 activity / silence 片段，每段附起止秒数和平均 dBFS。若原始包络超过 1200 窗，结果会返回 approximate=true 与 resolution_s，明确时间边界受降采样影响；片段数超过 128 时，列表有界并由 segments_truncated=true 标记。写入 audio 模态的短摘要会列出前 8 段时间线，并明确其余段数。该画像只表达声学活动，不代表语音、说话人或语义事件识别。

infer_audio_activity_period(features, silence_dbfs=-45, tolerance=0.2, min_fronts=4) 是纯特征映射函数，不访问设备或记忆。它从 RMS 包络中的可观测静音→活动前沿估计重复间隔；默认为至少 4 个前沿、3 个稳定间隔，周期至少是 4 个真实分析窗。条件不满足时返回 period_s=null 和具体 reason（包括 no_activity、no_observed_fronts、too_few_activity_fronts、intervals_not_stable、insufficient_resolution）；降采样包络保守拒绝估计。周期只表示声学活动重复间隔，不代表音乐节拍、语音韵律、事件身份或世界模型证据。

## 支持和边界

- WAV：支持 8/16/24/32-bit、1–8 声道未压缩 PCM，无外部依赖；
- 可选压缩输入：MP3、M4A/MP4、AAC、OGG/Opus、FLAC、AIFF、WMA、WebM；需本机 FFmpeg；
- 默认最多分析 300 秒、64 MiB PCM 数据；时长上限可由调用方调整，最大 3600 秒；
- window_ms 范围为 10–1000；silence_dbfs 范围为 -120–0；
- FFmpeg 输入仅接受本地文件或 bytes-like 数据，使用已识别容器的 demuxer，经 stdin pipe 解码；不启用网络协议、不使用 shell、不写临时文件；
- 压缩输入与解码 PCM 均最多 64 MiB，默认最多 300 秒（可调至 3600 秒），解码超时默认 90 秒（最高 600 秒）；
- 损坏/截断数据、空音频、未知格式、超预算或不可用 FFmpeg 都显式失败；
- 失败发生在 add_perception 之前，不留下半截记忆节点；
- FFmpeg 解码为单声道 8 kHz、16-bit PCM；结果标记 source_audio_format 与 decoded_from_compressed；原始压缩字节和 PCM 不落盘；
- 原始音频不持久化，结果只包含确定性声学特征；
- 能量包络最多 1200 个窗口点，静音区间最多返回 64 段，超限明确标记截断。

## 复跑

    python -X utf8 -m pytest tests/test_world_audio_perception.py -q --no-header  # 含真实 FFmpeg MP3 往返；FFmpeg 缺失时跳过

测试使用合成 WAV 覆盖声学边界与记忆闭环、稳定/不规则/不足活动周期及 11,025 Hz 实际窗时长换算；另有真实 FFmpeg MP3 往返与 mock 解码失败路径。
