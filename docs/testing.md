# 业务场景测试

测试围绕使用者需要完成的事组织：记忆能否在重启后继续使用，世界预测能否
接受真实观测反馈，描述能否变成正确的图像，训练产物能否交给下一次运行。
新增场景采用普通 pytest 函数、公有入口和直接断言；夹具只负责输入与资源生命周期。

## 可沿用的场景

| 文件 | 业务链 | 判据来自哪里 |
| --- | --- | --- |
| `tests/test_memory_world_scenarios.py` | 观测、因果证据入库 → 重开 SQLite → 召回与因果路径 | 温室事故的指定事实、两条因果边及其观测来源 |
| 同上 | 三帧小车运动 → 下一帧预测 → 真实继续/转向 → 验证与世界图更新 | 每帧一米的题设，以及单独提供的下一帧坐标 |
| 同上 | 模拟运行 → WAL 落盘 → 新实例回放 → 记忆载荷 → 从载荷恢复先验 | 预先指定的三步轨迹及其终点 |
| `tests/test_visual_scenarios.py` | 中文描述 → 图像/旋转 → 像素定位与属性读回 | 文字指定的颜色、大小和字面九宫格位置 |
| 同上 | `gen` 双主体渲染 → 仅给像素的主体测量 | 事先指定的数量、右上/左下位置和蓝色通道优势 |
| `tests/test_kernel_scenarios.py` | 合成传感器帧训练 → JSON 固化 → 重载 → 另一帧上的卷积 | 产物保留学习核及来源，并在序列化精度内保留使用行为 |

已有脑侧 M1/M2 和未观测预测回归也进入同一 pytest 收集入口。新增场景中，
固定输入与真实组件连用，断言描述可观察的结果；不会启动外部模型或使用私有素材。

## 运行

在仓根安装测试所需的公开依赖：

```bash
python -m pip install -e ".[full,dev]"
python -m pytest --collect-only -q
python -m pytest -q -m "not slow and not integration"
```

`python -m pytest -q` 收集并执行所有分组，配置没有默认排除项。
`--strict-markers` 使分组名称的拼写错误直接报错。

| 分组 | 命令 | 需要什么 |
| --- | --- | --- |
| 快速集 | `python -m pytest -q -m "not slow and not integration"` | numpy、Pillow、pytest；公开合成输入 |
| 训练集 | `python -m pytest -q -m slow` | 分层网络训练，运行时间较长 |
| 脑侧集成 | `python -m pytest -q -m integration` | 显式提供 dsh-memory 源码目录 |

`slow` 标在真正触发层级训练的测试上；解析、场景数据构造等同文件中的快速测试
仍进入快速集。`test_hex_search.py` 的所有测试共用 400 步训练夹具，因此整模块
标为 `slow`。外部依赖缺失产生有原因的 `SKIP`，不会算作通过。

`test_hex_cnn_upstream_asset` 使用原导出测试所需的 `data/img/0.png`，公开仓缺少
该素材时明确跳过。构造性 CNN 检查照常执行；公开合成图上的固化场景另外执行。
原有 `test_hex_cnn.py`、`test_wm_verify_unobserved.py`、`test_brain_store.py`
仍保留直接运行的脚本入口。

### 脑侧集成

准备可运行的 [dsh-memory](https://github.com/FuRongJun-1999/dsh-memory) 源码，
设置 `MDCG_BRAIN_PYTHONPATH` 为包含 `md_cg/` 和 `utf8_boot.py` 的目录，然后运行：

```bash
MDCG_BRAIN_PYTHONPATH=/path/to/dsh-memory python -m pytest -q -m integration
```

PowerShell：

```powershell
$env:MDCG_BRAIN_PYTHONPATH = (Resolve-Path ../dsh-memory).Path
python -m pytest -q -m integration
```

测试通过真实 MCP stdio 完成观测写入、场景重建和状态更新。数据根、辅助根
均使用临时目录；测试结束关闭进程。测试所需身份只作用于这份临时库。
配置目录存在但握手或业务断言失败时，测试会失败。

## 一个场景的写法

1. **先声明事实。** 写出使用者输入、初始世界、目标位置或持久化后应恢复的内容。
   数据要足够小，让读者能手工说明预期。
2. **经过真实流程。** 调用生产入口，让 SQLite、文件格式、渲染器等相关组件真实运行。
   使用 `tmp_path` 管理持久化产物；有关闭接口的资源用 `try/finally` 释放。
3. **用独立判据断言。** 预期来自场景规格、几何或外部观测；由直接断言显示实际差异。
   例如指定匀速轨迹后，下一步坐标应能手算；指定红色物体后，真实像素应出现在指定格子。
4. **检查交接结果。** 重开数据库、从磁盘读 WAL、重新加载 JSON 或读取图像属性，
   确认下一段业务拿到的东西仍有意义。

新测试不必创建统一基类或复刻生产算法。一个辅助函数在同文件的多个场景真正
需要它时再抽取。固定随机种子用于重现输入；时间语义通过已有 `clock` 注入口
指定；浮点容差应说明物理尺度或序列化精度。

对生成/验证组合尤其需要外部事实：只比较生成器返回的部件和同一次调用的日志，
不能证明图像画对了。像素位置、数量、颜色等应从真实图像读取，并与事先声明的
要求比较。回放也应与指定轨迹的终点比较，而非只调用系统自己的「一致」报告。

对产物持久化，写入前后的等价比较是业务契约：应覆盖下一次使用的行为。
核固化场景在另一幅图上比较重载前后的响应，并按 JSON 六位小数说明误差上限；
均匀传感器场还以「场值 × 权重和」手算响应。这个场景检验产物交接，
训练精度仍由训练集的学习判据检验。

## CI 与当前覆盖范围

`.github/workflows/tests.yml` 在 PR 和 main 推送上执行快速集：Linux Python 3.10、
3.13 以及 Windows Python 3.13。手动运行工作流并启用 `run_slow` 可运行训练集。
训练集失败会正常使该任务失败。脑侧集成由显式配置的环境执行，普通 CI 不下载
外部真源或访问在役记忆库。

当前覆盖从上述业务链扩展，尚未覆盖每个模块。快速 CI 通过表示所选契约通过；
模块覆盖率或测试数量不能代替业务能力验收。

### 已有失败

导出说明已经记录了 `tests/test_hex_text.py` 的两项失败（见
[body-export-v0.2 §四/§五](intake/body-export-v0.2/README.md)）：

- `test_spatial_detect_finds_single_object`：位置命中 6/16，未达到原判据。
- `test_multimodal_check_clean_vs_corrupt`：干净描述可接受 7/12，未达到原判据。

两项保留原断言、输入和阈值，仍以真实失败呈现。`slow` 是执行成本分组；
它同样包含当前通过的训练场景。复跑该模块：

```bash
python -m pytest -q tests/test_hex_text.py
```

后续增加测试时，优先选择尚未覆盖的正常工作流和有证据的业务回归。
每个新增断言都应能说明：哪种实际错误会被它发现，以及这会影响哪一步使用。
