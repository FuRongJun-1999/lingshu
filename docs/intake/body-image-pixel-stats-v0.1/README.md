# 身体图像像素统计 v0.1 · 新件登记

来源：本仓实现。单张已解码图像的确定性像素测量，不增加视觉 provider、语义识别或记忆能力。

## 范围

- `lingshu.vision.analyze_image_pixels(image)` 接收 Pillow `Image.Image`，返回 JSON-safe 尺寸、每通道 16 桶直方图、亮度摘要和 alpha 摘要。
- 只接受当前帧已解码图像：调用方须先执行 `image.load()`；未解码、关闭及多帧 `seek()` 后待解码状态直接拒绝，统计不触发流读取。
- 像素上限 4,194,304；超限直接拒绝，不缩放。
- alpha 像素按白底合成后统计可见 RGB；另报源 alpha 的不透明类别计数。亮度是 Pillow `convert("L")` 的 8-bit 编码空间统计，不声称物理/线性亮度。
- ICC/gamma 管理不额外改变；Pillow 自身负责源 mode 到 RGBA 的转换。
- 只接受内存图像对象；不读文件/网络、不写文件/数据库、不返回原像素、不调用模型。
- Pillow 仍是可选依赖，并在调用点导入；`import lingshu.vision` 不加载 Pillow/NumPy。

## 验证清单

- 合成 RGB 纯色和红/蓝块：颜色桶的边界、计数和总数精确。
- 合成灰阶：亮度范围、平均值和直方图确定。
- 合成 RGBA、LA 或调色板透明图：白底可见颜色与 alpha 类别分别正确。
- 超像素输入在 `convert` 前拒绝；非法输入有清楚类型错误。
- 固定输出有界、可 JSON 序列化；公开 package import 不预载 Pillow/NumPy。
- 不改 `perceive_image()`、provider 行为、能力状态、schema 或存储契约。
