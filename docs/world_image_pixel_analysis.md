# 本地图像像素统计 v0.1

`analyze_image_pixels(image)` 接受已经解码的 Pillow `Image.Image`，以固定且可复现的量化指标总结单张图片。它不调用视觉模型、不访问路径或网络、不保存图像，也不写入身体记忆。

## 统计口径

- 统计原图尺寸，不做缩放；`width`、`height`、`pixel_count` 和输入 `source_mode` 均在结果中报告。
- 像素数硬上限为 4,194,304。超限在 Pillow 转换/像素复制前拒绝；无隐式降采样。
- 可见颜色先转为 RGBA，再把 alpha 合成到不透明白底；R/G/B 统计该可见结果。每通道返回 16 桶直方图：桶 `v // 16`，因此最后一桶覆盖 240–255；每个通道计数和等于 `pixel_count`。
- 亮度使用合成后的 RGB 依 Pillow `convert("L")` 的 8-bit 编码空间转换口径，报告 min/max/mean/stddev 及同样的 16 桶直方图。结果是编码样本统计，不解释为线性光能量。
- alpha 摘要描述源图转换到 RGBA 后的 alpha：min/max/mean，以及完全透明、部分透明、不透明像素数。无 alpha 的模式自然报告所有像素 alpha=255。
- 不应用 ICC profile 或额外 gamma 校正；其他 Pillow 支持的模式按 Pillow 转成 RGBA 的默认模式转换规则处理。
- 输出只有固定结构和最多 64 个直方图整数，不含像素数组、路径、Pillow 对象或用户图像内容。

## 依赖与验证

Pillow 是可选依赖，仅调用统计函数时导入；只导入 `lingshu.vision` 不需要 Pillow 或 NumPy。调用时若 Pillow 未安装会明确报错。回归使用内存构造的纯色、色块、灰度与带透明度图片检查计数和均值，并覆盖尺寸预算、坏输入、JSON 序列化及轻量导入。
