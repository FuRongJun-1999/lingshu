# 身体视频帧变化统计 v0.1

本模块通过可选 FFmpeg 从本地短视频按固定 FPS 抽样，缩放/补边为固定 64×64 灰度帧，只返回相邻样本间归一化像素平均绝对差。它描述的是像素变化强度，不是场景识别、目标检测、动作/事件理解或视频记忆。

## 接口

```python
from lingshu.vision import analyze_video_frames

stats = analyze_video_frames(
    "clip.mp4", sample_fps=1.0, max_frames=120,
    max_duration_s=60, timeout_s=60,
)
```

返回的可序列化统计包括帧数、采样 FPS/间隔、采样网格时间、每对相邻帧的平均差值、总体均值和帧数预算截断标志。帧像素不返回、不写盘；函数不写记忆。

## 支持与资源边界

- 支持已识别本地 MP4/MOV、MKV/WebM、AVI、MPEG Program Stream 和 MPEG-TS 容器；也支持有相同签名的 bytes-like 输入。
- 需要本机可选 FFmpeg；仅调用 FFmpeg 子进程，不新增 Python 音视频依赖。
- 输入最多 64 MiB；默认最长分析 60 秒（最高 600 秒）、采样 0.1–10 FPS、最多返回 120 帧、每帧固定 64×64 灰度。
- 禁止网络 URL、UNC 和映射网络盘路径（在调用 isfile/open 之前拒绝；Windows 只接受 DRIVE_FIXED/DRIVE_REMOVABLE）。本地路径的各级组件在读取前检查 symlink/junction/reparse point，发现即拒绝。
- FFmpeg 使用固定 demuxer、首视频流、pipe-only、shell=False 与有界 worker；墙钟截止覆盖 stdin 写入、stdout 读取和等待；stderr 丢弃。内部和公共超时异常均不携带部分帧字节。
- 源视频解码开销仍受 FFmpeg、编码和解析器影响；时长、输入字节、输出帧数、输出内存和墙钟时间均有界。
- MP4/MOV 走 stdin 非 seekable pipe，因此仅接受 top-level moov atom 位于 mdat 前的 faststart 布局；使用常量内存 box 扫描并限制 top-level box 数；尾部 moov（包括其前带大 free atom 的文件）在启动 FFmpeg 前明确拒绝，提示 remux faststart。
- 时间戳使用 index / sample_fps 的采样网格，不是容器原生 PTS；不可用于精确音视频同步。
- fps filter 保留 0 秒的首个网格样本，因此比采样间隔更短但有一帧的视频仍返回一帧。
- SpacetimeMemoryEngine 的 modalities.video 继续为 False：帧统计不是视频语义感知，也不建立视频记忆节点。

## 复跑

```sh
python -X utf8 -m pytest tests/test_world_video_perception.py -q --no-header
```

测试以小型人工灰度帧验证 0/1 变化量与单帧行为；使用实际不消费 stdin 的子进程测试整体 deadline、Windows 路径 stub 测 UNC/映射盘预拒绝。若 FFmpeg 支持 MPEG-4 编码器，则真实测 faststart MP4、带 1 MiB free box 的尾 moov MP4 和 0.25 秒单帧短片。