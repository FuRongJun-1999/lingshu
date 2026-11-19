# 身体视频帧变化统计 v0.1 · 本仓新件登记

来源：本仓新实现；独立的身体侧视频基础统计，不复制外部模块。

新增件：
- lingshu/vision/video.py：可选 FFmpeg 解码、固定网格灰度抽样与相邻帧差统计；
- lingshu/core/core.py：analyze_video_frames 只读委托和能力声明；视频语义/记忆模态仍未宣告可用；
- tests/test_world_video_perception.py：mock 帧、错误/预算路径和可选真实 MP4 fixture；
- docs/world_video_frame_analysis.md：用法、时序口径和安全/隐私边界。

## 内容与隐私清单

- 固定输出仅 64×64 灰度帧；只返回 JSON 统计，不返回或持久化帧像素。
- 输入限已识别的本地视频格式或签名字节；最多 64 MiB、600 秒、120 样本帧；FFmpeg 仅走 stdin/stdout pipe、无 shell、无网络协议并受超时约束。
- 不加载 YOLO/CLIP、NumPy、Pillow，不新增 Python 依赖；可选 FFmpeg 缺失时显式拒绝。
- 统计只表示采样网格上的相邻灰度像素差；不能解释为镜头切分、物体/动作识别或事件语义。时间为采样网格而非媒体 PTS。
- 本件不拓展 add_perception 的 modality 白名单、不写记忆、也不把 modalities.video 改为 True。

## 验证面

- 手工小灰度帧：恒定帧差 0，黑→白差 1，单帧无相邻差；
- 限定 frame count、均值、网格时间与解码字节数；验证 shell=False、pipe-only、指定首视频流；
- 拒绝 URL、未知签名、输入超预算、FPS/帧数无效、FFmpeg 缺失/失败/超时；
- 可用时真实合成 MP4 抽帧，断言返回无源路径且无原始帧字节；
- 复跑：python -X utf8 -m pytest tests/test_world_video_perception.py -q --no-header。

范围不包括语义视频识别、目标检测、视频记忆存储、视频音轨转写/合并或实时摄像头。