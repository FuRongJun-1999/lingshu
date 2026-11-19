# 身体音频感知 v0.1

lingshu.vision.audio 提供一个不依赖 numpy/Pillow 的 PCM WAV 声学摘要入口，属于身体侧感知模块。它不是语音识别器：不转写内容、不推断说话人，也不把原始音频或输入路径写入灵枢记忆。

## Python 接入

SpacetimeMemoryEngine 提供 perceive_audio(path, ...)，也可直接调用 perceive_wav(engine, path, ...)：

    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    result = engine.perceive_audio("sample.wav")
    assert result["raw_audio_stored"] is False

结果包含时长、采样率、声道、位深、RMS/峰值 dBFS、活动比例、削波比例、有界能量包络、静音区间，以及摘要节点 id。摘要以 audio 模态、audio_summary 标签写入既有感知入口。

## 支持和边界

- 支持 8/16/24/32-bit、1–8 声道、未压缩 PCM WAV；
- 默认最多分析 300 秒、64 MiB PCM 数据；时长上限可由调用方调整，最大 3600 秒；
- window_ms 范围为 10–1000；silence_dbfs 范围为 -120–0；
- 压缩 WAV、损坏/截断数据、空音频、超出资源预算都会显式失败；
- 失败发生在 add_perception 之前，不留下半截记忆节点；
- 原始音频不持久化，结果只包含确定性声学特征；
- 能量包络最多 1200 个窗口点，静音区间最多返回 64 段，超限明确标记截断。

## 复跑

    python -X utf8 -m pytest tests/test_world_audio_perception.py -q --no-header

测试使用临时目录合成 WAV，覆盖静音/活动区间、立体声削波、四种 PCM 位深、边界拒绝和 audio 模态记忆闭环。
