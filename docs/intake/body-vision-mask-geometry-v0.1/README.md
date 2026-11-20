# 身体二值掩码几何 v0.1 · 新件登记

来源：本仓新实现。纯 Python 有界连通域，输入必须是调用方明确给出的二值掩码。

## 功能与边界

- 计算前景像素总数及 4/8 邻接连通域；按面积过滤并有界输出区域面积、半开 bbox、像素中心质心和边界接触状态。
- 不从图像颜色推断掩码，不识别物体、不填洞、不读图像文件，不调用模型、不写记忆。
- 输入最多 262,144 格，且每格严格 bool；矩形、参数和预算先校验，超限在 visited/queue 分配前拒绝。
- 返回 JSON-safe 有界摘要；组件输出最多 128 个，其余计入 omitted 数，不保存标签图或像素集合。

## 验证面

- 两个矩形、单像素、全空/全满、边界区域及环形孔洞。
- 对角点在 4/8 邻接下组件数量不同；面积过滤、同面积稳定顺序和最大组件截断。
- 非布尔 cell、不规则矩阵、非正尺寸、非法参数及超预算输入；验证超限先拒绝再分配。
- 输入不变、`json.dumps(..., allow_nan=False)`、不导入 NumPy/Pillow。
- 新增模块不修改 `perceive_image`、provider、存储 schema 或其他生成连通域实现。

## 复跑

```sh
python -m pytest tests/test_vision_mask_geometry.py tests/test_vision_lineart.py tests/test_doc_module_counts_guard.py tests/test_issue79_architecture_layers_guard.py tests/test_issue455_subprocess_encoding.py tests/test_issue80_intake_rerun_executable_guard.py tests/test_issue156_component_discovery.py tests/test_gate_python_floor_and_matrix.py tests/test_manifest_integrity.py -q --no-header
# 97 passed
python -X utf8 tools/intake_smoke_min.py --quiet
# import ok=90 fail=0 constructive=ok
```
