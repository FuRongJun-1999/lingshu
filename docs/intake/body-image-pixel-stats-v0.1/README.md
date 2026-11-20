# 身体图像像素统计 v0.1 · 新件登记

来源：本仓实现。单张由调用方解码的图像的确定性像素测量，不增加视觉 provider、语义识别或记忆能力。

## 范围

- `lingshu.vision.analyze_image_pixels(image)` 接收 Pillow `Image.Image`，返回 JSON-safe 尺寸、每通道 16 桶直方图、亮度摘要和 alpha 摘要。
- 像素上限 4,194,304；超限直接拒绝，不缩放。
- alpha 像素按白底合成后统计可见 RGB；另报源 alpha 的不透明类别计数。亮度是 Pillow `convert("L")` 的 8-bit 编码空间统计，不声称物理/线性亮度。
- 不额外应用输入附带的 ICC profile 或 gamma；源 mode 到 RGBA 的转换保留 Pillow 默认规则（`P` 模式的 `info["transparency"]` 透明索引会被计入），该转换路径可能使用 Pillow 内建 profile，故不承诺「绝对不含色彩管理」。
- 解码由调用方完成：只接受调用方提供的 `Image.Image`，只校验类型与尺寸，不要求预先 `load()`、不校验解码安全性（不是解码沙箱）；本函数不自行打开路径或访问网络，但若传入未加载的文件图对象，Pillow 的转换会读取其已打开的流。不写文件/数据库、不调用模型。
- 统计字段是内容派生数据（聚合计数与直方图由图像内容决定），不能承诺与图像内容无关；可具体承诺不返回原始像素数组、路径或输入附带的元数据（如 `icc_profile`、`filename`）。
- Pillow 仍是可选依赖，并在调用点导入；`import lingshu.vision` 不加载 Pillow/NumPy。

## 验证清单

- 合成 RGB 纯色和红/蓝块：颜色桶的边界、计数和总数精确。
- 合成灰阶：亮度范围、平均值和直方图确定。
- 合成 RGBA、LA 或调色板透明图：白底可见颜色与 alpha 类别分别正确。
- 超像素输入在 `convert` 前拒绝；非法输入有清楚类型错误。
- 固定输出有界、可 JSON 序列化；公开 package import 不预载 Pillow/NumPy。
- 不改 `perceive_image()`、provider 行为、能力状态、schema 或存储契约。
