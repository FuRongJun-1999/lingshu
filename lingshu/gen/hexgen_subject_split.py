"""**主体实例抽取**（R318，荣 2026-09-29 裁决「做主体的分割吧」后立项的第一增量；旁路新建）。

**为什么需要它**：#11 的余下 8 个域（count / composition / pose / view_dir / view_distance / occlusion /
dof / subject_priority）统一阻塞在「主体/部件求解器」上——它们都要**逐主体**的量，而现有读者只给
「簇」（`C.objects_of`）+ 一条保守计数规则（`C.count_objects`）；**重叠的物体在掩膜上本来就是同一个
连通域**（R279 实测：重叠簇里最大分量占比中位 1.000），几何切分在 R286 量否（凸性缺陷、1px 底）。
所以本模块的目标不是「再猜一个数」，而是**给出实例掩膜**，并把「可判 / 不可判」显式量化。

**白箱纪律**：纯标准库 + numpy、零 LLM、零链式法则、零训练；全部判据都在**源端可测**且**尺度无关**。

**三段式判据**（每一段都写出「为什么这样写」）：

  ① **前景 / 簇**：沿用读者的同口径（`C.fg_mask`，`TOL=40`；`C.cluster_of_pixels`，`d=6`）
     —— 与既有读数可比，且口径是同一条，不引入第二把尺子。
  ② **实例提议**＝**最终腐蚀**（ultimate erosion）：逐层腐蚀掩膜，取「**大分量个数最多**」那一层的分量质心。
     （R318 首版走的是「内切距离场取局部极大」，实测**两次翻车**并已弃用：① 距离场在**紧贴边界的子图**上
     没有外部可腐蚀 ⇒ 死循环打到上限；② 即便修好，「平台/脊上取 ≥ 局部极大」**天然多峰**——一块方块读出
     **16 个提议点**、`405` 三件读出 **36 个实例**。R279 量否的「DT 峰计数」是同一个病。最终腐蚀**不数峰**，
     只看「腐蚀到哪一层会分成几块」。**边界（如实登记）**：非凸形状的**臂**（星/心）也会分开 ⇒ 多出种子，
     真伪交由④的缝判据裁决。）
  ③ **切分**：前景像素按「到提议点的**最近**」归属（欧氏近似，非测地——如实登记为近似）。
  ④ **提议合并的判据（R320 起默认 `vocab`）**：相邻/邻近两件**该不该并**，由两个判据族裁决，亮度缝那一族
     两轮都量否（见下），R320 换成**词表吻合度**——**合并后的整体是否比拆开更像本词表 8 类形状**：
         `r(mask) = min over 8 类解析原型 的加权特征距离`（特征＝读者现成的 `C.shape_features`，其文档即
         「尺度无关 + 旋转不变」；权重＝**原型间特征标准差**，近常数维不参与；**判据全在源端**——
         原型来自自家 `shape_mask_v3`，**读者的判决不参与**，纪律 77）。判据＝`r(i∪j) < min(r(i), r(j)) − margin`。
     **R320 实测**：正/负对照干净、留出与自家**一项不动**、多 seed 误差总量 **45 → 35**（见下方 R320 段）。
     **两处机制性发现**：① **合并必须放在「全局一遍」而不是「簇内」**——诊断实测误差多数落在**簇**这一层
     （一个对象被前景掩膜断成多簇：盒 `406` 真值 1 读 2、`502` 真值 1 读 3、多 seed `s11` 真值 2 读 7），
     簇内合并**看不到这些对**；② 判据的**朝向无关**是必须的（`log_aspect` 取绝对值，见 `VOCAB_FEATS` 注释）。

  ⑤ **亮度缝判据（降档为诊断，`USE_SEAM=False`）**：R318/R319 两代统计量都**不可用**——
     `median`（带内中位色 ΔE）把细带不连续**抹平**（不同色重叠上恒 0.0）；`profile`（剖面最大跳变 ÷ 区内
     自身纹理）正对照好（硬阶跃 40.0、不同色重叠 8.0–8.33）但**无安全门**：真值 n=1 题面里「相邻区对」
     （**必属同一对象**）的假缝上界 **197.3** ≫ 真缝下界 8.0 ⇒ 门一开读数就掉（留出 15/19→12/19、
     多 seed 22/40→20/40）。根因＝缝的幅度与单件自身纹理**同量级**（「噪声底＝信号」族，同 R278/R286）
     ⇒ 按纪律 4 降档；`SEAM_STAT="profile"` 保留只作诊断（`_seam_profile_ratio` 文档串记全过程）。

**本增量实测（R318）**：
  * **自家渲染件**（逐对象参考掩膜＝精确真值）：实例数 **37/37 精确**、**IoU 均值 0.974**（标定 0.978 / 留出 0.971）、`IoU≥0.7` 100%；
  * **黑箱 37 条**：实例数精确 **24/37**（基线规则 `C.count_objects` 23/37）、±1 30/37（基线 30/37）；
  * **多 seed 40 张**（黑箱自身布局差异，含重叠）：精确 **22/40（55.0%）vs 基线 12/40（30.0%）**、
    ±1 **33/40（82.5%）vs 23/40（57.5%）** ⇒ 各 **+25.0pp**（本模块的主要增益来源）；
  * 两条负结果：**缝判据（现统计量）有害**（留出 11/19 vs 15/19）⇒ 默认关；**`MIN_CORE` 近惰性**：标定集精确 100→**8/18**、250/500→**9/18**，留出集三档同（**15/19**）⇒ 在**标定集**取 **250**（与 500 并列、取小）、登记为开关、不继续调参。

**误差桶（R319 实测，默认档 ⇒ 下一轮定靶用）**：精确 / 欠拆 / 过拆 ＝ 黑箱 37 条 **24 / 4 / 9**、
多 seed 40 张 **22 / 10 / 8**、自家 37 件 **37 / 0 / 0**。**主导误差是「过拆」**（R318 已量：非凸形状的臂、
自身纹理把一块切成多块），而缝通道两轮都救不了它 ⇒ 下一轮的靶是**过拆**（换「词表吻合度」判据），
`n_proposed`（提议数）与 `n_clusters`（簇数）一并输出，构成 `簇数 ≤ 实例数 ≤ 提议数` 的**不确定带**。

**R320 实测（词表吻合度判据，默认开）**：
  * **正/负对照（构造件，成对决策）**：该并的正例——星的两半 **2.355/2.551 → 0.040**、方块的两半
    **0.331/0.388 → 0.015** ⇒ **并** ✓；不该并的负例——两圆重叠（0.027/0.027 → **0.854**）、
    两**竖**矩形接触（0.628/0.023 → 1.436）、两星相邻（0.040/0.040 → **2.173**）⇒ **不并** ✓；
    **在 margin 0.0–0.30 上判定完全一致**（非刀锋门）。
  * **语料（`Σ|实例数−真值|`＝误差总量）**：标定集 **20 → 19**、留出集 **5 → 5**、自家 **0 → 0**、
    多 seed **45 → 35**（−22%，10 次合并）⇒ **留出与自家一项不动、多 seed 真实改善**。
    ⚠ **精度桶看不见它**：精确数（24/37、22/40）**一个字都没变**——这些误差件即使并掉几块仍不到真值
    ⇒ 若只看「精确/欠拆/过拆」三桶，本判据会显得**完全惰性**（R320 首版就栽在这，见 R320 段教训①）。
  * **两条如实登记的负结果**：① `MERGE_GAP`（允许的接触间隙）**近惰性**——只有标定集那 1 件受它影响
    （gap 0 与 4/12/24 之差），留出/自家/多 seed 三档不靠它；② **「按候选尺度选原型档（尺度阶梯）」被量否**：
    它修好了「三角原型自残留 0.46」却把别的档上的**近常数维**放大（两件整圆的残留 0.026 → **5.45**），
    实测会把「两圆重叠」这个**负例误并** ⇒ 回退为**固定参考 + 去掉 `n_peaks`**（R281 的结论：峰计数在
    离散掩膜上多计、亚像素峰显著度可用）。**边界（如实登记）**：原型自残留最高的是 `triangle` **0.333**
    （来自读者 `peak_prom` 的尺度敏感，属**读者特征**性质，本轮不改）。
  * **未引入过并**：`under` 桶在四档上都不增（标定 4 / 留出 0 / 自家 0 / 多 seed 10）。

**选型纪律**：阈值类量只在**标定集**（前 18 条 prompt）上选，判定在**留出集**（后 19 条）上报（同 R272）。
"""
import math
import os
import sys

import numpy as np

from . import hexgen_c1_real as C          # noqa: E402
from . import hexgen_self_source as S      # noqa: E402  只读复用：8 类**解析原型**（词表）

SIZE = 512
MIN_CORE = 250               # **最小核面积**（px）：腐蚀后的大分量需达此面积才算「一个主体核」。
                             #   ⚠ 消融实测：**标定集**精确 100→8/18、250/500→9/18；**留出集**三档同（15/19）
                             #   ⇒ 在标定集取 250（与 500 并列取小）、按纪律**登记为开关**、不继续调参。
USE_SEAM = False             # **缝判据开关（默认关；R319 降档为诊断）**：两轮消融都**有害**——
                             #   黑箱**留出集**件数精确 15/19（关） vs **11/19**（开·median） vs **12/19**（开·profile）；
                             #   多 seed 22/40（关） vs 20/40（开）。根因＝缝的幅度与单件自身纹理**同量级**
                             #   （假缝上界 197.3 ≫ 真缝下界 8.0）⇒ 通道不可分，不是门的问题（见模块头根因段）。
SEAM_STAT = "profile"        # 缝判据统计量：`profile`＝R319 新统计量（默认，诊断用） | `median`＝R318 旧统计量（量否）
SEAM_MIN = 0.05              # 旧统计量的门（仅 `SEAM_STAT="median"` 时生效）
SEAM_MIN_PROFILE = 3.0       # 新统计量的门（**剖面最大跳变 ÷ 区内自身纹理**；标定集上选）
SEAM_BAND = 3                # 接触带宽度（px）
MIN_PX_SPLIT = 200           # 参与提议的簇面积下限（更小的簇不拆）


def _erode(m, r=1):
    return C._dilate(m, r) if r <= 0 else _erode_n(m, r)


def _erode_n(m, r):
    """方形结构元腐蚀 r 次（与 `C._dilate` 对偶：用膨胀的补集实现）。"""
    return ~C._dilate(~m, r)


EDT_PAD = 64                 # 距离场补零留白（≥ 词表最大半径 51 ⇒ 腐蚀一定收敛到空）


def _ultimate_erosion_seeds(sil, min_core=60, max_layer=120):
    """**最终腐蚀（ultimate erosion）种子**：逐层腐蚀掩膜，取「大分量个数最多」的那一层的分量质心。

    **为什么不用「距离场取局部极大」（R318 首版实测两次翻车）**：
      ① 距离场在**紧贴边界的子图**上没有外部可腐蚀 ⇒ 我首版的 `~dilate(~m)` 永远不空（打到 600 上限、
         EDT 全 600）⇒ 必须**补零留白**（`EDT_PAD`）；
      ② 即使修好，「平台/脊上取 ≥ 局部极大」**天然多峰**——实测一块方块读出 **16 个提议点**、
         `405` 的三件方块读出 **36 个实例**。R279 当年量否的「DT 峰计数」是同一个病。
    最终腐蚀换了取法：**不数峰，看「腐蚀到哪一层会分成几块」**（经典形态学分裂准则），
    只用**掩膜**、只用两个源端量（层数与小分量门 `min_core`），与题面无关。
    **边界（如实登记）**：非凸形状（星/心）的**臂**在腐蚀中也会分开 ⇒ 会多出种子；
    这一步因此**只产生提议**，真伪由**缝判据**裁决（无缝则并回）。"""
    m = sil.copy()
    best = None
    for level in range(1, max_layer + 1):
        m = ~C._dilate(~m, 1)
        if not m.any():
            break
        lab, sizes = C.components(m)
        big = [i for i in range(len(sizes)) if sizes[i] >= min_core]
        if big:
            if best is None or len(big) > len(best[1]):
                best = (level, [(int(np.mean(np.nonzero(lab == i)[0])),
                                 int(np.mean(np.nonzero(lab == i)[1])), int(sizes[i])) for i in big])
    if best is None:
        ys, xs = np.nonzero(sil)
        return [(float(ys.mean()), float(xs.mean()), float(sil.sum()))]
    return [(float(y), float(x), float(a)) for y, x, a in best[1]]


def _seam_ratio(img, lab, i, j, bg_lab, band=SEAM_BAND):
    """相邻两块 i/j 的**缝判据**：接触带两侧内环的中位色 ΔE ÷ d_obj。返回 (比值, 两侧像素数)。"""
    mi = lab == i
    mj = lab == j
    if not mi.any() or not mj.any():
        return 0.0, 0, 0
    touch = C._dilate(mi, band) & C._dilate(mj, band)
    if not touch.any():
        return 0.0, 0, 0
    bi = touch & mi
    bj = touch & mj
    if bi.sum() < 8 or bj.sum() < 8:
        return 0.0, int(bi.sum()), int(bj.sum())
    ci = np.median(img[bi], axis=0)
    cj = np.median(img[bj], axis=0)
    d_obj = float(np.linalg.norm(C.rgb_to_lab(ci) - bg_lab)) if bg_lab is not None else 0.0
    if d_obj <= 1e-6:
        return 0.0, int(bi.sum()), int(bj.sum())
    return float(np.linalg.norm(C.rgb_to_lab(ci) - C.rgb_to_lab(cj)) / d_obj), int(bi.sum()), int(bj.sum())


def _seam_profile_ratio(img, lab, i, j, bg_lab=None, band=SEAM_BAND, nbins=24, min_bin=6):
    """**缝判据（R319 新统计量）**：沿 A→B 轴的**剖面最大跳变** ÷ 两侧**区内自身纹理**。

    **为什么必须换统计量（R318 的负结果）**：旧统计量取「接触带内环的**中位色** ΔE ÷ d_obj」——
    中位是**抗离群**的统计量，而缝恰恰是**一条细带上的离群** ⇒ 它把要测的东西**抹平**了
    （黑箱留出集：开 11/19 < 关 15/19，开比关更差）。
    新统计量：把两块**一起**投影到 A→B 轴上、对垂直方向取中位得到 1-D 强度剖面；
        `seam = max|Δ剖面|（中央接触区） ÷ (中位|Δ剖面|（两侧内部区） + 1)`
     ① 用 **max** ⇒ 细带上的跳变不会被平均掉；② 分母是**区内自身纹理**（同单位、同尺度、同归一）
     ⇒ 尺度无关、与颜色无关（不需要 `d_obj`）；③ 只在**连续**的剖面段上差分（缺箱跳过）。
    返回 `(比值, 像素数, 像素数)`（与旧统计量同形，便于同一处调用）。"""
    mi, mj = lab == i, lab == j
    if int(mi.sum()) < 8 or int(mj.sum()) < 8:
        return 0.0, 0, 0
    yi, xi = np.nonzero(mi)
    yj, xj = np.nonzero(mj)
    ci = (float(yi.mean()), float(xi.mean()))
    cj = (float(yj.mean()), float(xj.mean()))
    ay, ax = cj[0] - ci[0], cj[1] - ci[1]
    n = math.hypot(ay, ax)
    if n < 4.0:
        return 0.0, 0, 0
    ay, ax = ay / n, ax / n
    both = mi | mj
    ys, xs = np.nonzero(both)
    t = (ys - ci[0]) * ay + (xs - ci[1]) * ax                 # 沿 A→B 轴的位置
    inten = img[ys, xs].mean(axis=1)                           # 亮度（缝在亮度上最直接）
    bins = np.clip((t / n * nbins).astype(int), 0, nbins - 1)
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = bins == b
        if int(sel.sum()) >= min_bin:
            prof[b] = float(np.median(inten[sel]))
    idx = np.where(~np.isnan(prof))[0]
    if idx.size < 6:
        return 0.0, 0, 0
    d = np.abs(np.diff(prof[idx]))
    pos = (idx[:-1] + 0.5) / float(nbins)
    central = (pos > 0.30) & (pos < 0.70)
    inner = (pos <= 0.25) | (pos >= 0.75)
    if not central.any() or not inner.any():
        return 0.0, 0, 0
    med_inner = float(np.median(d[inner]))
    mx = float(np.max(d[central]))
    return mx / (med_inner + 1.0), int(both.sum()), int(both.sum())


_SEAM_FN = {"median": _seam_ratio, "profile": _seam_profile_ratio}


# ==================== R320 · 提议合并的判据：**词表吻合度**（源端解析原型） ====================
#   动机（R319 结论）：亮度缝**不可分**（假缝上界 197.3 ≫ 真缝下界 8.0）⇒ 按纪律 100「该改的是判据族，
#   不是门」。过拆的形态是「**一块被切成了几块**」，而**本词表 8 类形状**恰好给出一个与像素强度无关的
#   判据面：**合并后的整体是否比拆开更像词表里的一类**。
#   **判据全在源端**：8 类**自家解析原型**（`S.shape_mask_v3`）的特征中心 vs 候选掩膜的特征；特征用读者
#   现成的 `C.shape_features`（其文档即「尺度无关 + 旋转不变」，两侧同一把尺子）；**读者的判决
#   （`C.predict`）不参与**（纪律 77：判决量严禁参与选型）。
MERGE_STAT = "vocab"         # 提议合并判据：`vocab`（R320 起**默认**）| `none`（R318/R319 默认＝不并）| `seam`（R318/R319 量否）
MERGE_MARGIN = 0.0           # `vocab` 的余量门（**标定集**上选：gap=4 时标定集 Σ|Δ| **19@0.0** vs 20@0.15/0.30
                             #   ⇒ 取 0.0；**源端配对对照**给的安全带是 [0, 0.86]（最差正例的改善量 0.448 ＜
                             #   最好负例的间距 0.86）⇒ 0.0 在带内、非刀锋门）
MERGE_GAP = 4                # 合并的**邻近前提**（两掩膜最大间隙 px）。**标定集上选**：Σ|Δ| 20@gap0 vs
                             #   **19@gap∈{4,12,24}**（并列 ⇒ 取最小 4）；留出/自家/多 seed 三档**逐位相同**
                             #   ⇒ 作用只在标定集那 1 件（间隙 ≤4px 的断口），登记为**近惰性**开关
VOCAB_MIN_SPREAD = 0.02      # 逐维参与门槛：跨原型标准差 < 此值 ⇒ 该维**近常数**、不含词表信息（源端判据，非拟合）
VOCAB_REF = 128              # 原型参考半径（固定）——**同尺度比同尺度**的替代做法见下
#   ⚠ **为什么不用尺度阶梯、而是「固定参考 + 去掉 `n_peaks`」**：特征号称尺度无关，但**离散掩膜上的峰
#   **计数**不无关**——同一个解析三角，rad=128 的 `n_peaks` 是 **3**、rad=51 的是 **5**（台阶锯齿多计；
#   R281 早量过「合成六边形 10 vs 解析 6」）。试过「按候选有效半径选原型档（尺度阶梯）」：它虽修好了三角，
#   却在别的档上把**近常数维**放大（实测两件整圆的残留从 0.026 涨到 **5.45** ⇒ 该并的负例被误并），
#   故回退。最终采用 R281 的结论：**峰计数量去掉**（`n_peaks` 不入判据），保留亚像素的 `peak_prom`/`peak_flat`。
VOCAB_LADDER = (12, 17, 24, 34, 48, 68, 96, 136, 192)   # 保留（诊断/复现用），生产路径不用
VOCAB_FEATS = ("extent", "abs_aspect", "solidity", "compactness",
               "peak_prom", "peak_flat") + tuple("harm%d" % k for k in range(1, 9))
#   ⚠ `abs_aspect`（取绝对值）是**必须的**：读者的 `log_aspect` 是**带符号**的 `log(w/h)`，而方位对我们
#   无意义。实测两件**竖矩形**（各读 −0.775）与原型 `rectangle`（+0.775）相距 **6.3σ** ⇒ 若用带符号版本，
#   两块「本身很像词表」的矩形会因为**朝向**被判成「都不像」、从而被并成一块（**误并**）。
#   取绝对值后朝向不参与（半个方块 1.47 → 与矩形原型近邻），误并面随之关掉。

# ==================== R321 · 提议切分的判据：**词表吻合度的镜像**（先于合并遍执行） ====================
#   靶（R320 登记的下一步②）：**欠拆桶**（标定 4 / 多 seed 10）——「多件并成一簇」。合并判据吃
#   「碎片不像词表、并入主体后骤降」；切分是反方向：**一件自己不像词表、而能切成两块各更像词表则切**。
#   判据＝`r(a) + r(b) + SPLIT_MARGIN < r(M)`（M＝整块，a/b＝PCA 主轴 1-D 2-means 的两半），
#   `r`＝同一个词表残留（与合并判据**共用一把尺子**）。两判据**互斥**：若同一对 (a,b) 同时成立，
#   则 r(a)+r(b) < r(M) < min(r(a), r(b)) ⇒ min < 0，不可能 ⇒ 合并遍**永远不会撤掉**切分遍的成果。
#   **R321 诊断的机制发现**（先量再调）：欠拆实例普遍「**自己就像词表**」（rM 0.44–1.38）——「多件
#   并成一簇」在特征空间里落在**原型邻域内**，并不表现为「离词表远」；切成两半反而**更不像**
#   （r(a)+r(b) 1.6–2.7 > rM）⇒ 和式判据的触发面天然**窄**（四语料 14 个欠拆实例只触发 2 个：
#   标定 #10 rM=1.150→0.819、#17 rM=6.778→6.343，都是 rM 被拉高的特例）。**碎片是全部假触发源**：
#   碎片残留天然高（multi #0 inst5 面积 3021 / rM 3.116 被误切 7→8）⇒ 加**对象尺度下限**挡住
#   （碎片归合并遍管，不归切分遍）。**multi 欠拆桶的另一半不是本判据族的靶**（如实登记）：两条子
#   通道——整幅前景失败（s44/s55 面积 262144，面积守卫另案）与「重叠件低残留融合」（#3/#4/#9/#34/#37，
#   rM 0.44–1.38、切分和 1.6–2.6 不触发）——后者需要与词表无关的**重叠/凹点**信号，见台账。
SPLIT_STAT = "vocab"         # 提议切分判据：`vocab`（R321 起**默认**）| `none`（R318–R320 行为＝不切）
SPLIT_R_MIN = 0.9            # 触发前提：整块残留 ≥ 此值才尝试切分。**标定集上选**：带地板后
                             #   rmin∈[0.5, 1.1] × margin∈[0, 0.3] **整片平坦最优**（标定 Σ18/(10,2,6)）；
                             #   1.2 起 #10（rM=1.150）不再触发 ⇒ Σ 回 19。取 0.9（#10 上方余量 0.25、
                             #   距 self 实例残差上界 0.35 有 0.55 带）
SPLIT_MARGIN = 0.2           # 接受门：r(a)+r(b)+margin < r(M)。标定 0.0–0.3 平坦；#10 的改善量 Δ=0.331，
                             #   margin 0.3 时余量仅 0.031 太贴边 ⇒ 取中间 0.2（余量 0.131）
SPLIT_DEPTH = 2              # 递归深度上限（3 件簇需两层：1|2 再拆 2）
SPLIT_MIN_AREA = 8000        # **对象尺度下限**（候选实例须 ≥ 此面积才尝试切分）：锚＝词表标称半径 51
                             #   的整圆面积 π·51²≈8171。标定集上 (3021, 14242] 任意取值同效——
                             #   它挡掉的是**全部**假触发（无地板时低 rmin 的标定 over 5→7 全是 <8000px 碎片）
SPLIT_MAX_AREA_FRAC = 0.5    # 整幅前景失败（面积 > 0.5·SIZE²，multi s44/s55＝262144）不切——另一通道
_VOCAB_CACHE = {}


def _vfeat(f):
    """把读者特征字典投影到**词表判据面**：`log_aspect`（带符号）→ `abs_aspect`（取绝对值，朝向不参与）。"""
    g = dict(f)
    g["abs_aspect"] = abs(float(f["log_aspect"]))
    return g


def proto_features(ref=96):
    """（兼容入口）**某一档尺度**上的 8 类解析原型特征。生产路径不直接用它——`vocab_residual` 会按
    候选掩膜的有效半径**选档**（见 `_ladder_ref`）。"""
    return _proto_at(ref)[0]


def vocab_scale(ref=96):
    """（兼容入口）某一档的逐维尺度＝**原型间的标准差**（源端尺度，与读者无关）；**近常数维不参与**。
    ⚠ 为什么近常数维必须排除（R282 的教训）：跨原型散度只有 0.003 的维（实测 `harm1`）一旦当分母，
    会把那些维放大成硬约束（R282 里 MAD 尺度把 `harm1/harm7` 放大 **200×**、直接压垮形状分类）；
    「近常数维不含词表信息」是**定义**上的判断，故这里用源端绝对阈值 `VOCAB_MIN_SPREAD`，不是拟合参数。"""
    return _proto_at(ref)[1]


def _proto_at(ref):
    """**某一档尺度**的 8 类解析原型特征 + 逐维尺度 + 参与维（一次缓存；确定性）。"""
    key = "proto@%d" % ref
    if key not in _VOCAB_CACHE:
        rows = {}
        for sh in S.SHAPES8:
            m = S.shape_mask_v3(sh, SIZE, SIZE // 2, SIZE // 2, ref) > 0.5
            rows[sh] = _vfeat(C.shape_features(m)[0])
        sc = {}
        for k in VOCAB_FEATS:
            v = float(np.std([rows[sh][k] for sh in S.SHAPES8]))
            sc[k] = v if v >= VOCAB_MIN_SPREAD else 0.0
        _VOCAB_CACHE[key] = (rows, sc, [k for k in VOCAB_FEATS if sc[k] > 0.0])
    return _VOCAB_CACHE[key]


def _ladder_ref(area):
    """候选掩膜的**有效半径** `sqrt(area/π)` 落到尺度阶梯的哪一档（**同尺度比同尺度**）。"""
    reff = math.sqrt(max(1.0, float(area)) / math.pi)
    return VOCAB_REF          # 生产路径用**固定参考**（见 VOCAB_REF 注释：阶梯在本语料上被量否）


def vocab_residual(mask):
    """候选掩膜到**词表**的残留：`min over 8 原型` 的加权特征距离（0＝与词表里某一类同形）。
    掩膜太小（<8px）或特征不可算 ⇒ 返回 `None`（**判据不适用 ⇒ 保守不并**，如实登记）。"""
    if mask is None or int(mask.sum()) < 8:
        return None
    got = C.shape_features(mask)
    if not got:
        return None
    f = _vfeat(got[0])
    protos, sc, used = _proto_at(VOCAB_REF)
    best = None
    for sh in S.SHAPES8:
        acc = n = 0.0
        for k in used:
            acc += ((f[k] - protos[sh][k]) / sc[k]) ** 2
            n += 1.0
        if n <= 0.0:
            continue
        d = math.sqrt(acc / n)
        best = d if best is None else min(best, d)
    return best


def _crit_of(merge, use_seam):
    """判定用哪个合并判据：显式 `merge` 优先；只给了老开关 `use_seam=True` ⇒ `seam`；否则模块默认。"""
    if merge is not None:
        return merge
    if use_seam:
        return "seam"
    return MERGE_STAT


def _split_two(mask):
    """PCA 主轴投影上的**确定性** 1-D 2-means（初值取投影两端极值；平局归近端）。
    退化（点太少 / 投影无宽度 / 一侧空 / 一侧 < `MIN_PX_SPLIT`）⇒ `None`（保守不切）。"""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n < 2 * MIN_PX_SPLIT:
        return None
    pts = np.stack([ys, xs], 1).astype(np.float64)
    q = pts - pts.mean(0)
    cov = (q.T @ q) / n
    _w, vec = np.linalg.eigh(cov)
    t = q @ vec[:, int(np.argmax(_w))]
    c1, c2 = float(t.min()), float(t.max())
    if not (c2 > c1):
        return None
    asg = None
    for _ in range(64):
        asg = np.abs(t - c1) <= np.abs(t - c2)
        if asg.all() or not asg.any():
            return None
        n1, n2 = float(t[asg].mean()), float(t[~asg].mean())
        if n1 == c1 and n2 == c2:
            break
        c1, c2 = n1, n2
    a = np.zeros_like(mask)
    a[ys[asg], xs[asg]] = True
    b = mask & ~a
    if a.sum() < MIN_PX_SPLIT or b.sum() < MIN_PX_SPLIT:
        return None
    return a, b


def _vocab_split_one(mask, r_min, margin):
    """单层切分判定（R321 判据面，全在**源端**；读者的判决不参与，纪律 77）。
    过全部前提 ⇒ 返回 (a, b)；任一不过 ⇒ `None`（保守不切）：
    ① 面积窗 [`SPLIT_MIN_AREA`, `SPLIT_MAX_AREA_FRAC`·SIZE²]（碎片归合并管 / 整幅前景失败另案）；
    ② 触发前提 `r(M) ≥ r_min`；③ 可二分；④ 两半残留可算且 `plausibly_object`；
    ⑤ 接受门 `r(a) + r(b) + margin < r(M)`。"""
    area = int(mask.sum())
    if area < SPLIT_MIN_AREA or area > SPLIT_MAX_AREA_FRAC * SIZE * SIZE:
        return None
    r_m = vocab_residual(mask)
    if r_m is None or r_m < r_min:
        return None
    parts = _split_two(mask)
    if parts is None:
        return None
    ra, rb = vocab_residual(parts[0]), vocab_residual(parts[1])
    if ra is None or rb is None:
        return None
    if not (C.plausibly_object(parts[0]) and C.plausibly_object(parts[1])):
        return None
    if ra + rb + margin >= r_m:
        return None
    return parts


def _vocab_split_pass(out, img, r_min, margin, depth):
    """全局切分一遍：对每个实例递归尝试（深度 ≤ `depth`）。返回 (新实例表, 切分事件数)。"""
    events = 0

    def walk(m, d):
        nonlocal events
        parts = None if d <= 0 else _vocab_split_one(m, r_min, margin)
        if parts is None:
            return [_instance_of(img, m)]
        events += 1
        pieces = []
        for p in parts:
            pieces.extend(walk(p, d - 1))
        return pieces

    new_out = []
    for m, _inst in out:
        new_out.extend(walk(m, depth))
    return new_out, events


def _instance_of(img, m):
    """由掩膜构造实例记录（面积/包围盒/质心/中位色/Lab）；`fused`/`seam` 由合并阶段回填。"""
    iy, ix = np.nonzero(m)
    inst = {"area": int(m.sum()),
            "bbox": (int(iy.min()), int(ix.min()), int(iy.max()), int(ix.max())),
            "centroid": (float(iy.mean()), float(ix.mean())),
            "median_rgb": np.median(img[m], axis=0),
            "fused": False, "seam": None}
    inst["lab"] = C.rgb_to_lab(inst["median_rgb"])
    return (m, inst)


def split_subjects(img, tol=C.TOL, d=6, min_px=C.MIN_PX, seam_min=None,
                   use_seam=USE_SEAM, seam_stat=None, jitter=0,
                   split=None, split_r_min=None, split_margin=None, split_depth=None,
                   merge=None, merge_margin=None, merge_gap=None):
    """**主体实例抽取**。返回 dict(`instances`, `diag`)。

    `instances`：[(掩膜 bool, 诊断 dict), ...]；诊断含 `area/bbox/centroid/median_rgb/lab/`
    `n_merge_events`（缝判据并回去几次）/`fused`（该实例是否由「无判据的并」得到）。
    R321 起管线为**切分遍 → 合并遍**（判据互斥 ⇒ 合并不会撤掉切分的成果）：`split="vocab"`
    开切分遍（`split_r_min`/`split_margin`/`split_depth` 覆盖模块默认），诊断多
    `split_crit`/`n_split_events`。
    `jitter`：只供守门做**确定性**检查用（对提议点做固定抖动并比较结果是否同构）。"""
    img = np.asarray(img, np.float64)
    if seam_min is None:
        seam_min = SEAM_MIN if (seam_stat or SEAM_STAT) == "median" else SEAM_MIN_PROFILE
    merge_margin = MERGE_MARGIN if merge_margin is None else merge_margin
    merge_gap = MERGE_GAP if merge_gap is None else merge_gap
    raw, _dist = C.fg_mask(img, tol=tol)
    bg = C.bg_color(img)
    bg_lab = C.rgb_to_lab(bg)
    lab_d, n_d = C.cluster_of_pixels(raw, d)
    out = []
    diag = {"n_clusters": int(n_d), "n_merge_events": 0, "n_fused": 0, "n_no_split": 0,
            "n_proposed": 0}   # `n_proposed`＝**提议总数**（未并）⇒ 与 `n_instances` 一起构成**不确定带**
    for c in range(n_d):
        sel = (lab_d == c) & raw
        if sel.sum() < min_px:
            continue
        sil = C.fill_holes(sel)
        ys, xs = np.nonzero(sil)
        area = int(sil.sum())
        if area < MIN_PX_SPLIT:
            pts = [(float(ys.mean()), float(xs.mean()), float(area))]
            diag["n_no_split"] += 1
        else:
            seeds = _ultimate_erosion_seeds(sil, min_core=MIN_CORE)
            if jitter:
                seeds = [(y + (jitter if k % 2 else -jitter), x + (jitter if k % 3 else -jitter), v)
                         for k, (y, x, v) in enumerate(seeds)]
            pts = seeds or [(float(ys.mean()), float(xs.mean()), float(area))]
            if len(pts) == 1:
                diag["n_no_split"] += 1
        diag["n_proposed"] += len(pts)
        #   切分（欧氏最近）
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        best = np.full((SIZE, SIZE), np.inf, np.float32)
        labm = np.full((SIZE, SIZE), -1, np.int32)
        for i, (py, px, _v) in enumerate(pts):
            dd = (yy - py) ** 2 + (xx - px) ** 2
            upd = sil & (dd < best)
            labm[upd] = i
            best[upd] = dd[upd]
        n_lab = len(pts)
        for i in range(n_lab):
            m = labm == i
            if m.sum() < min_px:
                continue
            out.append(_instance_of(img, m))
    # ---------------- 全局切分一遍（R321 · 合并判据的镜像，**先于**合并；两者互斥 ⇒ 无振荡） ----------------
    scrit = SPLIT_STAT if split is None else split
    if scrit not in ("none", "vocab"):
        raise ValueError("unknown split criterion: %r" % (scrit,))
    diag["split_crit"] = scrit
    diag["n_split_events"] = 0
    if scrit != "none":
        sr_min = SPLIT_R_MIN if split_r_min is None else split_r_min
        smargin = SPLIT_MARGIN if split_margin is None else split_margin
        sdepth = SPLIT_DEPTH if split_depth is None else split_depth
        out, diag["n_split_events"] = _vocab_split_pass(out, img, sr_min, smargin, sdepth)
    # ---------------- 全局合并一遍（判据 `crit` = none | seam | vocab） ----------------
    #   ⚠ **R320 的关键改动：合并从「簇内」升到「全局」**。R320 诊断实测：误差多数落在**簇**这一层
    #   （一个对象被前景掩膜断成多簇：盒 `406` 真值 1 读 2、`502` 真值 1 读 3、`105` 真值 2 读 4、
    #   多 seed `s11` 真值 2 读 7）——簇内合并**看不到这些对**；而词表判据恰好适用：碎片到词表的残留
    #   实测 1.8~6.5（不像任何一类），并入主体后骤降 ⇒ 该并。邻近前提＝`MERGE_GAP`（两掩膜最大间隙 px；
    #   0＝只并**接触**的两件，与历史口径同）。**不变量**：`none` 时一行不动（旧读数可复跑）。
    crit = _crit_of(merge, use_seam)
    diag["merge_crit"] = crit
    if crit != "none" and len(out) > 1:
        band = SEAM_BAND if crit == "seam" else max(1, int(math.ceil(max(0.0, merge_gap) / 2.0)))
        lab_all = np.full((SIZE, SIZE), -1, np.int32)
        for k, (m, _i) in enumerate(out):
            lab_all[m] = k
        parent = list(range(len(out)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        dil, res, vals = {}, {}, {}

        def dil_of(i):
            if i not in dil:
                dil[i] = C._dilate(out[i][0], band)
            return dil[i]

        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if not (dil_of(i) & dil_of(j)).any():
                    continue                  # 邻近前提（`MERGE_GAP`＝0 ⇒ 只并接触的两件）
                if crit == "seam":
                    r, ni, nj = _SEAM_FN[seam_stat or SEAM_STAT](img, lab_all, i, j, bg_lab)
                    ok = bool(ni and nj and r < seam_min)
                else:
                    for k in (i, j):
                        if k not in res:
                            res[k] = vocab_residual(out[k][0])
                    if res[i] is None or res[j] is None:
                        continue              # 判据不适用（掩膜太小/特征不可算）⇒ 保守不并
                    r = vocab_residual(out[i][0] | out[j][0])
                    if r is None:
                        continue
                    ok = bool(r < min(res[i], res[j]) - merge_margin)
                vals[(i, j)] = r
                if ok:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                        diag["n_merge_events"] += 1
        groups = {}
        for i in range(len(out)):
            groups.setdefault(find(i), []).append(i)
        merged = []
        for _root, members in sorted(groups.items()):
            if len(members) == 1:
                merged.append(out[members[0]])
                continue
            m = out[members[0]][0].copy()
            for k in members[1:]:
                m |= out[k][0]
            inst = _instance_of(img, m)[1]
            inst["fused"] = True
            diag["n_fused"] += 1
            inst["seam"] = max((vals.get((min(a, b), max(a, b)), 0.0)
                                for a in members for b in members if a != b), default=None)
            merged.append((m, inst))
        out = [x for x in merged if x[1]["area"] >= min_px]
    diag["n_instances"] = len(out)
    return {"instances": out, "diag": diag}


# ==================== 参考掩膜（自家渲染件的精确真值） ====================

def reference_masks(tup, zone_calib=None, pal=None, size=SIZE, rad_frac=0.10):
    """自家渲染件的**逐对象参考掩膜**（精确真值）：同一个渲染管线，逐个对象单独画一次。
    返回 [(掩膜 bool, 元数据 dict), ...]，元数据含 `shape/cx/cy/rad/color/pattern`。"""
    from . import hexgen_self_source as S
    n, shape, color, pattern, cells = tup
    pts, rad = S.place_objects(cells, n, max(8, int(size * rad_frac)), size,
                               shape=shape, zone_calib=zone_calib)
    out = []
    for cx, cy in pts:
        cov = S.shape_mask_v3(shape, size, cx, cy, rad)
        out.append((cov > 0.5, {"shape": shape, "cx": int(cx), "cy": int(cy), "rad": int(rad),
                                "color": color, "pattern": pattern}))
    return out


def iou(a, b):
    inter = int((a & b).sum())
    uni = int((a | b).sum())
    return inter / uni if uni else 0.0


def match_instances(insts, refs):
    """**贪心匹配**（按 IoU 降序）：返回 [(inst_idx, ref_idx, iou), ...] 与未匹配项。
    `insts`/`refs` 均为掩膜列表（取 `instances`/`reference_masks` 的第一列）。"""
    pairs = []
    for i, a in enumerate(insts):
        for j, b in enumerate(refs):
            v = iou(a, b)
            if v > 0:
                pairs.append((v, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_i, used_j, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, v))
    return out, sorted(set(range(len(insts))) - used_i), sorted(set(range(len(refs))) - used_j)
