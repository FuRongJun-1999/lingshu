# MV 制作插件（`wm_bridge.factory.film.mv`）

> **一句话**：输入一份 `brief.json`，输出一支完整 MV ——
> **视频 + 音轨 + 字幕 + 清单 + 视听真值裁定**，全程零 LLM、逐字节可复现。

影视层 `film/` 立的是**契约与判据**（P0/P1）。本插件是那份"可判定"的**执行器**（P2–P6）：
它把契约变成片子，再把片子**回读成事实**交给 P1 裁决。执行器**不给自己打分**。

---

## 一、快速开始

```bash
cd bridges/blender
export WM_LINGSHU_ROOT=/path/to/lingshu          # 上游仓根（见 factory/_deps.py）

# 1) 产出整支 MV（默认简报：夜行 · 24s / 960x540 / 24fps）
PYTHONPATH=. python3 -X utf8 -m wm_bridge.factory.film.mv.make --jobs 8

# 2) 只看判据（不重渲，秒级）——判据口径改了也不必重渲全片
PYTHONPATH=. python3 -X utf8 -m wm_bridge.factory.film.mv.make --judge

# 3) 冻结态验收（复算清单 / 确定性 / 判据不变式 / 字幕 / 契约）
PYTHONPATH=. python3 -X utf8 -m wm_bridge.factory.film.mv.make --accept

# 4) 小体积预览 GIF
PYTHONPATH=. python3 -X utf8 -m wm_bridge.factory.film.mv.make --preview

# 5) 探针（只渲 8 帧）
PYTHONPATH=. python3 -X utf8 -m wm_bridge.factory.film.mv.make --max-frames 8 --size 320x180 --jobs 4
```

产物落在 `$MV_OUT`（缺省 `./mv_out`，**不写进仓库**）。

---

## 二、制作流程（七步）

> 每一步都写清 **输入 → 做什么 → 产出 → 可核对读数**。
> 这七步就是"从一句话到一支片子"的全部；没有任何一步靠模型即兴发挥。

### 0. 创意简报（`brief.json`）—— LLM 只在**这一步**出现

| | |
|---|---|
| 输入 | 人的创意意图（题材/情绪/歌词/时长） |
| 做什么 | 编译成一份**声明式**简报：`title / duration_s / fps / tempo / key / style / sections / lyrics / scene` |
| 产出 | `brief.json`（本包附 `briefs/night_walk.json` 可直接跑） |
| 为什么 | **意图与执行分离**：简报是唯一"模糊"的产物；此后每一步都是确定性的算术 |

### 1. P3 乐谱与音轨（`_det/audio.py`）

| | |
|---|---|
| 做什么 | 按 `tempo.bpm/meter/offset` 用**闭式**算拍点 `beats[i]=offset+i·60/bpm`；按 `sections` 编排低音线/和弦垫/鼓组；段落能量随 `kind` 变化（intro 稀疏 → chorus 全开 → bridge 收） |
| 产出 | `music.wav`（44.1k/16bit/立体声）+ `wmmv.score/1`（拍点/强拍/段落/能量包络） |
| 可核对 | 峰值与软限幅（实测 **−0.92 dBFS**，无削顶）；副歌能量 **1.88×** 主歌；包络长度 `= round(duration/hop)+1`；同简报两次 → **逐字节同 sha256** |
| 契约硬约束 | 拍点严格递增、强拍 ⊆ 拍点、段落无缝且末段止于 `duration_s`、包络长度由 hop 唯一决定 |

### 2. P4 音画绑定（`_det/timeline.py::build_bind`）

| | |
|---|---|
| 做什么 | 能量分档 → 景别 / ASL（平均镜长）/ 运动量；`section_map` 给出每段落的默认景别与偏好机位；卡点规则 `on=downbeat`、容差 1 帧 |
| 产出 | `wmmv.bind/1` |
| 可核对 | `energy_map.bands` 覆盖 [0,1] 无缝；低能量 ↔ 长 ASL + 宽景别，高能量 ↔ 短 ASL + 中近景 |

### 3. P5 镜头表（`_det/timeline.py::plan_shot_table`）

| | |
|---|---|
| 做什么 | **骨架优先**：全片铺满、≥8 镜；每镜给出 `camera.path`（≥2 控制点、`t` 覆盖 [0,1]）+ `move_kind`（23 项闭集）+ `compose`/`light`；**所有切点由构造取自 downbeat 帧**；相邻镜头 `move_kind` 必变、景别跨级、机位夹角 ≥30°、**全片机位在动作轴线同一侧** |
| 产出 | `wmfilm.shot/1` × N |
| 可核对 | 实测 **10 镜 / 9 切点 / E1 最大偏差 0.000 帧**；相邻机位夹角 37.8°–43.8°；方位角全负（同一侧） |

### 4. P5 剪辑与文本（`_det/timeline.py::build_edit / build_text`）

| | |
|---|---|
| 做什么 | 镜头首尾相接成时间轴（**闭区间**：`b.start == a.end + 1`）；每个切点挂 `alignment`（`at_frame/beat_s/offset_frames`）；歌词按**段落出现次序**配对（chorus 出现两次各挂各的），CPS/行宽/行数/安全区由契约层硬拦 |
| 产出 | `wmmv.edit/1` + `wmmv.text/1` + `mv.srt` |
| 可核对 | 切点偏差全 0；CPS 最大 **7.5**（≤20）；6 条歌词一条不丢；字幕落在 title safe 0.80 内 |

### 5. P2 场景与取景（`_det/scene.py` + `_det/camera.py`）

| | |
|---|---|
| 做什么 | **建模** `mesh_gen`（球 96×48、肢体 48 环放样锥形管、发丝扫掠、手指成管、碑体倒角、灯杆旋转体+扫掠弯臂）→ 注册进 `mesh3d.MESH_LIBRARY` ⇒ `World3D.render()` 自动走**真 z-buffer**；**表演** `skeleton3d`（`human_skeleton` + `apply_walk`）逐帧摆 14 个部件；**物理** `physics.projectile_state` 预计算轨迹表；**着色** `shade_paint`（`studio` 光学模型 + 逐顶点法线光滑着色） |
| 取景 | `compose.coverage_target`（**包围盒面积占比**）→ 目标高度占比 `t=sqrt(c·a/F)` → 解距离；瞄准点绑主体包围盒中心 + `headroom_ratio` 解算纵向偏移 `Δ=h·(0.5−(0.5−r)/t)` + `subject_anchor` 沿相机右向量偏移 |
| 产出 | 逐帧 `World3D` + 逐帧 `Camera3D` |
| 可核对 | 场景 **130,708 面**、**26/26 水密**、法线朝外；实测高度占比与目标逐镜吻合（0.600/0.604、0.851/0.847…）；头 room ≈7–9%；`smooth` 开/关像素确实不同 |

### 6. §六 真值回读（`_det/oracle.py`）

| | |
|---|---|
| 做什么 | 从**渲出来的像素**回读：主体覆盖率（把"只有主体"的世界单独渲一遍数非键色像素）、屏幕包围盒、剪影 ΔL、亮度直方图 p05/p50/p95、色温估计、实际机位/焦距 |
| 产出 | 逐帧真值条目（进 `truth.json`） |
| 为什么 | **真值外部性**：`coverage_target` 是意图，`coverage_actual` 是事实；判据只吃事实 |

### 7. P1 判据裁决 + P6 出口（`_det/gates.py` + `_det/encode.py`）

| | |
|---|---|
| 做什么 | 把执行器产物装配成 P1 的输入束 `sc = {shots, edit, score, axis, truth, frame_size}`，交给 **`film/_det/gate_all`** 裁决（K1–K5 + E1 + 登记的 DEFER 族）；再编码母版 |
| 产出 | `wmfilm.oracle/1`（`oracle.json`）+ `mv.avi` + `mv.mp4` + `mv_preview.gif` + `manifest.json` |
| 可核对 | 判据票型（ACCEPT/BLINDSPOT/DEFER）；AVI 可回读（frames/size/fps）；清单逐件 sha256 |

---

## 三、产物

| 文件 | 内容 |
|---|---|
| `mv.avi` | 视频：**手写 RIFF/MJPG**（纯 Python + PIL，不依赖 ffmpeg） |
| `mv.mp4` | 交付形态：H.264 + AAC **含音轨**（探测到 ffmpeg 才产出，缺则**跳过而不失败**） |
| `music.wav` | 音轨：44.1k/16bit/立体声，逐字节确定 |
| `mv.srt` | 字幕：歌词（CPS/行宽/安全区经契约层硬拦） |
| `mv_preview.gif` | 预览：抽帧 + 缩放，体积可控 |
| `manifest.json` | 清单：每个产物 sha256 + 字节数 + 契约摘要 + 判据票型 |
| `oracle.json` | 视听真值裁定 `wmfilm.oracle/1` |
| `truth.json` | 逐帧回读证据（`--judge` 时落边界帧） |
| `score/bind/edit/text.json`、`shots/*.json` | 四份 + N 份契约文档 |
| `frames/*.png` | 逐帧画面（逐像素可复跑） |
| `acceptance.json` | 冻结态验收读数 |

---

## 四、判据对位（复用 P1，不重写）

| 判据 | 吃什么 | 本插件怎么供 |
|---|---|---|
| K1 轴线一致 | 声明机位 + **事实轴线** | 轴线 = 主体首末帧**真实世界位移**（两点定线） |
| K2 视线匹配 | 事实 `subject_gaze_screen` | **未回读 ⇒ 恒 BLINDSPOT**（§7.2.1 登记的契约缺口，不猜） |
| K3 屏幕方向 | 事实 `subject_motion_screen` | 几何口径 `sign(axis · camera_right)`（见下"口径表"） |
| K4 30° 规则 | 声明机位 | 相邻同主体镜头首帧夹角 |
| K5 景别跳跃 | 事实 `subject_bbox_screen` | 切点边界帧的包围盒高度 |
| E1 卡点对齐 | 声明切点 + 乐谱强拍 | `alignment` 的 `offset_frames` |

---

## 五、口径表（每一条都是被实测逼出来的，别照抄结论不看理由）

1. **帧号 = 闭区间**。`edit` 要求 `b.start == a.end + 1`，`text` 要求第二条 cue 的 `start >` 第一条的 `end`。
   ⇒ 执行器内部半开区间 `[start, end)`，**只在落契约时 −1**。
2. **世界 = Y-up，P1 口径 = Z-up**。`film/_det/scene.UP=(0,0,1)`（对齐 `geo/spec.py`），
   而渲染世界是 Y-up（`Object3D.size` 的 h 走 y、`_ground_y()` 按 y 贴地、`mesh_gen` 声明 Y 轴向上）。
   `gates.to_film_vec` 用**绕 x 轴 +90° 的正交旋转** `(x,y,z)→(x,−z,y)`（det=+1）做桥：
   保持 up 映射、夹角与**叉积符号** ⇒ 判据结论不变而输入束合规。**不要**用换轴（反射），那会翻手性。
3. **`Camera3D.from_look_at` 的默认 `up=(0,0,1)` 是陷阱**：近水平视线下它会让相机右向量变成 ≈ −y，
   画面**滚转约 90°、头落到脚下方**。一律显式 `up=(0,1,0)`；回归门**必须用离轴机位**才敏感
   （机位在 x=0 看向 x=0 平面时两种 up 等价 ⇒ 假绿）。
4. **`coverage_target` 是包围盒面积占比**，不是剪影像素覆盖率。按剪影标定会把镜头怼到主体溢出画面
   （实测相机离主体 0.6 m、**头被切掉**）。映射：`t = sqrt(c·a/F)`，再用**实测剪影包围盒高度占比**精修。
5. **`headroom_ratio` 的符号**：`Δ = h·(0.5 − (0.5−r)/t)`；写反会让主体整体下沉（实测 8% → **31%**）。
6. **K3 不能用逐帧像素位移**：执行器把瞄准点绑在主体上（避免出画）⇒ 主体屏幕净位移≈0，
   只剩步态抖动（峰峰 20~24 px、周期≈10 帧），符号由步态相位决定 = 抛硬币。
   故 `subject_motion_screen` 取**几何** `sign(axis · camera_right)`（= 观众看到的方向：主体不动、**背景在动**）。
7. **光滑着色的代价**：`mesh_smooth=True` 是**逐三角形成本**（上游非 flat 分支对每个三角形建 257 项 LUT）
   ⇒ 实测 flat **0.96 s** vs smooth **9.09 s**/帧，且**几乎与分辨率无关**。13 万面本身不是瓶颈。
   全片 576 帧按**帧块并行**（每帧是 `(shot, frame)` 的纯函数 ⇒ 并行不改变像素）。
8. **`smooth` 需要顶点法线**：`normals = mesh.normals if (smooth and mesh.normals) else None`
   ⇒ 网格不带 `MeshGeometry.normals` 时**静默退化为平面着色**。本包 26/26 件都补齐了，并有"开/关像素必须不同"的对照门。
9. **并行分块要用闭区间步进**：`f = g + 1`。写成 `f = g` 会让最后一块原地死循环
   （实测 RSS 爬到 6.5 GB 被 OOM 杀掉）。
10. **摘要必须 64 位**：P0 的 `_digest_str` 规定"形如十六进制就必须是 64 位 sha256"。

---

## 六、硬边界

| 边界 | 本插件如何守 |
|---|---|
| B1 不改 `lingshu/world/` | 只 import，不写；上游 `git status` 干净（26 个 world 文件 mtime 未变） |
| B2 不把 bpy 带进 lingshu | 执行器不 import bpy |
| B3 原版插件零改动 | 只在 `factory/film/mv/` 新增 |
| B4 契约层纯标准库 | `film/*.py` 与本包 `__init__` 可**无 PIL/numpy** 导入；执行器 `_det/*` 才依赖渲染所需依赖 |
| 依赖不写死路径 | 走 `factory/_deps.py`：`WM_LINGSHU_ROOT` → 仓内候选 → `DepError`（**明确失败**） |

---

## 七、已知缺口（如实登记，不粉饰）

1. **K2 恒 BLINDSPOT**：未回读 `subject_gaze_screen`（§7.2.1 登记的契约缺口）。
2. **K3 在"相机完全跟拍"时没有像素信号**：本包用几何口径补上（见口径 6）；若评审认为
   `subject_motion_screen` 必须是像素实测，则该字段应改为缺省、K3 落 BLINDSPOT。
3. **判据只实现 K 族 + E1**（架构已裁 D5）；C/M/A/P 四族由 P1 显式 `DEFER` 登记（实测 29 条）。
4. **画幅固定 16:9 横屏**（已裁 D4）；竖屏不在射程。
5. **`mv.mp4` 非逐字节确定**（H.264 编码器多线程/版本差异）；确定性以 AVI/帧序列/WAV/清单为准。
6. **GIF 时间粒度是厘秒**：24fps 请求 42 ms、盘上实为 40 ms（格式固有限制）。
7. **单机内存**：`--jobs` 过大（本机 12）会 OOM；960×540 建议 `--jobs 8`（实测 worker ≈250 MB/个）。

---

## 八、复现性

```bash
# 两次跑 → 逐字节比对（确定性声明）
MV_OUT=/tmp/a ... --max-frames 24 --size 320x180 --jobs 4
MV_OUT=/tmp/b ... --max-frames 24 --size 320x180 --jobs 4
sha256sum /tmp/a/music.wav /tmp/b/music.wav /tmp/a/mv.avi /tmp/b/mv.avi
```

* 每帧是 `(shot, frame)` 的**纯函数**；`--jobs` 只改变**谁**来算，不改变**算什么**。
* 随机性只出现在"颗粒/手持"等处，且**全部由帧号/镜头 id 决定**（无 `time`、无未播种 `random`）。
* `--judge` 从**已存帧**重跑判据 ⇒ 判据口径演进不必付一次全片渲染。
