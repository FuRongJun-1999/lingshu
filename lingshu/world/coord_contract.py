# -*- coding: utf-8 -*-
"""坐标口径的**唯一真源** · `wm.coord-frame/1`（第 4 轮优化 #4）

病史
-------------------------------------------------------------------------------
同一套"边界钳制 + 两位小数量化"在世界模型侧被**复制了 5 份**，
且各自长出不同分支（零方向是否量化 y、非有限坐标怎么处置、步长是否按剩余距离封顶）：

  | # | 站点 | x/z clamp | 量化 | y | 零方向短路 | 非有限 | 步长封顶 |
  |---|---|---|---|---|---|---|---|
  | 1 | `world_model.py::_apply_move` | `[0.5, size-0.5]` | round2 | 直通 | 三分量都量化 | 不查 | 无 |
  | 2 | `scene_simulator.py::step` | 同上 | round2 | 直通 | （速度 0 时自然不动） | **显式拒绝** | 无 |
  | 3 | `world_learner.py::_apply_move` | 同上 | round2 | 直通 | 三分量都量化 | 不查 | 无 |
  | 4 | `spacetime_consistency.py::_apply_move_at` | 同上 | round2 | 直通 | 三分量都量化 | **显式拒绝** | **`min(speed,dist)`** |
  | 5 | `voxel_world.py::step` | 同上 | **无** | **地面约束** | 无 | 不查 | 无 |

"复制 5 份"本身不是最糟的；最糟的是**第 5 份已经悄悄分化**（不量化、y 有地面约束），
而**没有任何判据**会在第 6 份出现时出声。本模块做两件事：

  ① 把 x/z 边界语义收成**一个**实现（`clamp_axis` / `clamp_xz` / `apply_move`），
     各站点的**差异一律显式命名成参数**，不再藏在各自的函数体里；
  ② 提供 `scan_clamp_copies()` —— 扫"**margin clamp**"模式，
     新出现的拷贝会让守卫转红，而不是让它悄悄住进代码库。

为什么用 `max(M, min(S - M, v))` 这个模式
-------------------------------------------------------------------------------
这正是本条语义的特征形状：**下界与上界减去的余量是同一个常数**
（正则里用**反向引用** `\\1` 钉住）。
它天然不会误伤概率钳制：`max(0.0, min(1.0, p))` 的上下界常数不同 ⇒ 不匹配。
"""
from __future__ import annotations

import ast
import math
import os
import re

COORD_CALIBER = "wm.coord-frame/1"

#: 边界余量（米）：实体中心离世界边缘至少留这么远
BOUND_MARGIN = 0.5
#: 量化位数（本体口径 = 两位小数）
QUANT = 2

# ---- 零方向短路策略（显式命名，不再藏在各函数体里）
ZERO_ROUND_ALL = "round_all"      # 三分量都量化（本体 / world_learner / spacetime）
ZERO_XZ_ONLY = "xz_only"          # 只量化 x/z，y 原样（P3 的 world 帧）

# ---- 非有限坐标策略（`#347`：clamp 表达式遇 NaN 会**静默**抬成边界值）
NF_NONE = None                    # 不检查（本体 / world_learner）
NF_REJECT_ROUND_ALL = "reject_round_all"    # 返回量化后的**原位置**（spacetime）
NF_REJECT_XZ_ONLY = "reject_xz_only"        # 只量化 x/z、y 原样（P3 的 world 帧）

SPEC = {
    "caliber_id": COORD_CALIBER,
    "semantics": ("x/z 边界钳制到 `[margin, size - margin]`（margin=%.1f），y **直通**"
                  "（bridge 实测 AXIS_MAP `(x,y,z)→(x,z,−y)` det=+1 ⇒ **y 是高度**）；"
                  "x/z 量化到 %d 位小数" % (BOUND_MARGIN, QUANT)),
    "why_y_passthrough": "地面/高度由世界自己管（voxel_world 另有地面约束），坐标契约不越权",
    "divergences_named": {
        "zero_shortcut": [ZERO_ROUND_ALL, ZERO_XZ_ONLY],
        "on_nonfinite": [NF_NONE, NF_REJECT_ROUND_ALL, NF_REJECT_XZ_ONLY],
        "step_cap": "步长按到目标的剩余距离封顶（spacetime #249）",
    },
    "hard_rules": [
        "本模块是 x/z 边界语义的**唯一**实现；各站点只许调用，不许再写一份。",
        "跨模块的**差异**必须写成显式参数，不许靠「这个函数里恰好是这么写的」。",
        "`scan_clamp_copies()` 的白名单 = 本模块自身；新拷贝会被守卫打红。",
        "改本模块的算式等于改**所有**站点的读数 ⇒ 必须同时跑域内零回归（trace_sha256 90/90）。",
    ],
}


def clamp_axis(v, lo, hi):
    """单轴边界钳制。**算式与本体逐字一致**（`max(lo, min(hi, v))`）。"""
    return max(lo, min(hi, v))


def clamp_xz(x, z, size, margin=BOUND_MARGIN):
    """把 (x, z) 钳到 `[margin, size - margin]`。

    ⚠ 算式必须与本体一字不差：`max(margin, min(size - margin, v))`。
    任何"顺手改成 `min(max(...))` 或先量化再钳"都会改掉域内 trace。
    """
    hi = size - margin
    return (clamp_axis(x, margin, hi), clamp_axis(z, margin, hi))


def apply_move(pos, speed, d, size, margin=BOUND_MARGIN, quant=QUANT,
               step_cap=None, on_nonfinite=NF_NONE, zero_shortcut=ZERO_ROUND_ALL):
    """按方向移动一步。**默认参数下的算式与本体 `_apply_move` 逐字一致**。

    `step_cap`       非 None 时步长按 `min(speed, step_cap)` 封顶（spacetime #249）
    `on_nonfinite`   非有限坐标的处置策略（`#347`：不许静默抬成边界值）
    `zero_shortcut`  零方向短路的量化范围（见 ZERO_* 常量）
    """
    q = (lambda v: round(float(v), quant)) if quant is not None else float
    if d == (0.0, 0.0, 0.0):
        if zero_shortcut == ZERO_XZ_ONLY:
            return (q(pos[0]), float(pos[1]), q(pos[2]))
        return tuple(round(float(v), quant if quant is not None else 6) for v in pos) \
            if quant is not None else tuple(float(v) for v in pos)
    step = speed if step_cap is None else min(speed, step_cap)
    nx = pos[0] + d[0] * step
    nz = pos[2] + d[2] * step
    if on_nonfinite is not NF_NONE and not (math.isfinite(nx) and math.isfinite(nz)):
        if on_nonfinite == NF_REJECT_XZ_ONLY:
            return (q(pos[0]), float(pos[1]), q(pos[2]))
        return tuple(round(float(v), quant if quant is not None else 6) for v in pos) \
            if quant is not None else tuple(float(v) for v in pos)
    lo, hi = margin, size - margin
    return (q(clamp_axis(nx, lo, hi)), float(pos[1]), q(clamp_axis(nz, lo, hi)))


# ------------------------------------------------------------------ 拷贝扫描
# 第 5 轮：从"三级简单绑定"升级为**仓库级符号表 + 常量折叠 + 调用点分析**。
#   新增能力（用户点名的两处漏洞）：
#     Cap-1 **跨文件**：`import` / `from … import … as …` 解析到**值**，按**值相等**比对
#           ⇒ `from .a import M as _M` + `max(_M, min(size - M, v))` 这种"结构不等、
#             值相等"的写法不再漏（此前只比结构）。
#     Cap-2 **经函数参数**：边界算式写在**调用点**、`max/min` 在共享 helper 里 ——
#           此前完全看不见。现在扫调用点：若某实参是 `X - M`、另一实参值与 `M` 相等，
#           就判定"**边界算式被搬到了调用点**"。
#   ⚠ **不能把"复用共享 helper"本身当成拷贝**：真实树里 `clamp_axis`/`clamp_xz` 就是
#     为此而存在（8 个参数化 clamp 定义里 7 个是 `max(-1,min(1,…))` 这类固定界，
#     语义不同）。故分档：**世界边界语义 （判死）** / **固定界（只报）**。

#: 正则档：`max(M, min(X - M, …))`。⚠ 只用来报"可疑"；**判死靠 AST 档**。
CLAMP_RE = re.compile(r"max\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*min\(\s*[^,()]*-\s*\1\s*,")
#: 第二档「可疑」：同行同时出现 `max(` / `min(` / 边界量词 / `0.5`。只报给人看。
SUSPECT_RE = re.compile(r"max\(.*min\(|min\(.*max\(")
SUSPECT_WORDS = ("size", "bound", "limit", "extent", "width", "height")

#: **共享实现**的名字（调用点分析要认得出"被复用的那一个"）
CANONICAL_HELPERS = ("clamp_axis", "clamp_xz")
#: 共享实现所在文件 —— 边界算式的**唯一**合法落点
CANONICAL_FILE = "lingshu/world/coord_contract.py"
#: 「边界算式 → 调用共享 helper」这个组合的**合法落点**：{`<file>::<函数>`: 允许处数}
CANONICAL_CALL_SITES = {CANONICAL_FILE + "::clamp_xz": 1}


def _is_builtin(call, name):
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == name)


def _norm(node):
    """结构等价（忽略位置）—— 值无法求得时的**兜底**判等。"""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _modname_of(rel):
    p = rel[:-3] if rel.endswith(".py") else rel
    parts = p.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _pkg_of(rel):
    mod = _modname_of(rel)
    if rel.endswith("__init__.py"):
        return mod
    return mod.rsplit(".", 1)[0] if "." in mod else ""


def _assign_env(node):
    """`node` 子树里 `NAME = <expr>` 的简单绑定（一层）。"""
    env = {}
    for st in ast.walk(node):
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)):
            env.setdefault(st.targets[0].id, st.value)
        elif (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                and st.value is not None):
            env.setdefault(st.target.id, st.value)
    return env


class _Repo:
    """**仓库级**符号表：一次解析全部 .py，供跨文件解析与调用点分析使用。"""

    def __init__(self, root, overrides=None):
        self.root = root
        self.files = {}
        self.by_mod = {}
        overrides = overrides or {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "node_modules", ".venv")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root).replace("\\", "/")
                try:
                    src = overrides[rel] if rel in overrides else open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                assigns, imports, funcs = {}, {}, {}
                for st in tree.body:                       # 只看**模块级**
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                assigns.setdefault(t.id, st.value)
                    elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        if st.value is not None:
                            assigns.setdefault(st.target.id, st.value)
                    elif isinstance(st, ast.ImportFrom):
                        base = self._abs_from(rel, st)
                        for al in st.names:
                            imports[al.asname or al.name] = (base, al.name)
                    elif isinstance(st, ast.Import):
                        for al in st.names:
                            imports[(al.asname or al.name).split(".")[0]] = (al.name, None)
                    elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funcs[st.name] = st
                info = {"rel": rel, "src": src, "tree": tree, "modname": _modname_of(rel),
                        "assigns": assigns, "imports": imports, "funcs": funcs}
                self.files[rel] = info
                self.by_mod.setdefault(info["modname"], rel)

    def find_mod(self, tmod):
        """把 import 里的模块名解析到**本仓库的文件**（含两级兜底）。

        兜底是必需的：`sys.path` 把包目录挂上去时 `import a` 指的是 `pkg/a.py`，
        而路径推出的模块名是 `pkg.a` ⇒ 只按全名查会**整片漏掉跨文件场景**（本轮实测）。
        兜底按"**末段名唯一**"或"**路径后缀唯一**"匹配；**有歧义（多个候选）就不猜**。
        """
        if tmod in self.by_mod:
            return self.by_mod[tmod]
        tail = tmod.split(".")[-1]
        cands = [r for r, m in self.by_mod.items() if m.split(".")[-1] == tail]
        if len(cands) == 1:
            return cands[0]
        suf = "/" + tmod.replace(".", "/") + ".py"
        cands = [rel for rel in self.files if rel.endswith(suf)]
        return cands[0] if len(cands) == 1 else None

    @staticmethod
    def _abs_from(rel, node):
        mod = node.module or ""
        if not node.level:
            return mod
        base = _pkg_of(rel)
        for _ in range(node.level - 1):
            base = base.rsplit(".", 1)[0] if "." in base else ""
        return (base + "." + mod).strip(".") if mod else base

    @staticmethod
    def local_imports(node, rel):
        """**函数内**的 import（很常见：为省启动成本把 import 放进函数）。

        ⚠ 只索引模块级 import 会整片漏掉这类跨文件场景（本轮实测：M18 变异把
        `from .coord_contract import BOUND_MARGIN as _M` 写在函数里，老实现抓不到）。
        """
        out = {}
        for st in ast.walk(node):
            if isinstance(st, ast.ImportFrom):
                base = _Repo._abs_from(rel, st)
                for al in st.names:
                    out[al.asname or al.name] = (base, al.name)
            elif isinstance(st, ast.Import):
                for al in st.names:
                    out[(al.asname or al.name).split(".")[0]] = (al.name, None)
        return out

    def value(self, expr, rel, env=None, cls_env=None, depth=0, imps=None):
        """求表达式的**常量值** ⇒ `(value|None, kind)`；`kind ∈ const/name/unknown`。

        `name` 表示"是个符号（名字/属性）"但值没折出来 —— 分档上仍算**变量档**。
        """
        env = env or {}
        cls_env = cls_env or {}
        imps = imps if imps is not None else ((self.files.get(rel) or {}).get("imports") or {})
        if expr is None or depth > 8:
            return None, "unknown"
        if isinstance(expr, ast.Constant):
            v = expr.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, "unknown"
            return float(v), "const"
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
            v, k = self.value(expr.operand, rel, env, cls_env, depth + 1, imps)
            if v is None:
                return None, k
            return (v if isinstance(expr.op, ast.UAdd) else -v), k
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub,
                                                               ast.Mult, ast.Div)):
            a, _ka = self.value(expr.left, rel, env, cls_env, depth + 1, imps)
            b, _kb = self.value(expr.right, rel, env, cls_env, depth + 1, imps)
            if a is None or b is None:
                return None, "unknown"
            try:
                if isinstance(expr.op, ast.Add):
                    return a + b, "const"
                if isinstance(expr.op, ast.Sub):
                    return a - b, "const"
                if isinstance(expr.op, ast.Mult):
                    return a * b, "const"
                return (a / b), "const"
            except ZeroDivisionError:
                return None, "unknown"
        if isinstance(expr, ast.Name):
            if expr.id in env:                                   # 函数/局部
                return self.value(env[expr.id], rel, env, cls_env, depth + 1, imps)
            if expr.id in cls_env:                               # 类属性（裸名）
                return self.value(cls_env[expr.id], rel, env, cls_env, depth + 1, imps)
            info = self.files.get(rel) or {}
            if expr.id in imps:                                  # ★ Cap-1：跨文件（含函数内 import）
                tmod, attr = imps[expr.id]
                if attr is not None:
                    trel = self.find_mod(tmod)
                    if trel:
                        tinfo = self.files[trel]
                        if attr in tinfo["assigns"]:
                            return self.value(tinfo["assigns"][attr], trel, {}, {},
                                              depth + 1, tinfo["imports"])
                return None, "name"
            if expr.id in (info.get("assigns") or {}):           # 模块级
                return self.value(info["assigns"][expr.id], rel, env, cls_env, depth + 1, imps)
            return None, "name"
        if isinstance(expr, ast.Attribute):
            base = expr.value
            info = self.files.get(rel) or {}
            if isinstance(base, ast.Name):
                if base.id in imps:                               # ★ mod.ATTR 跨文件
                    tmod, attr0 = imps[base.id]
                    if attr0 is None:
                        trel = self.find_mod(tmod)
                        if trel and expr.attr in self.files[trel]["assigns"]:
                            return self.value(self.files[trel]["assigns"][expr.attr],
                                              trel, {}, {}, depth + 1,
                                              self.files[trel]["imports"])
                if base.id in cls_env:                            # self.M（类属性）
                    v, k = self.value(cls_env[base.id], rel, env, cls_env, depth + 1, imps)
                    return v, k
            return None, "name"
        return None, "unknown"

    def same_bound(self, a, b, rel, env=None, cls_env=None, imps=None):
        """两个"界"是否同一**值**（值优先，结构兜底）—— Cap-1 的关键判据。"""
        va, _ = self.value(a, rel, env, cls_env, 0, imps)
        vb, _ = self.value(b, rel, env, cls_env, 0, imps)
        if va is not None and vb is not None:
            return abs(va - vb) <= 1e-12
        # 只要**不是两侧都解出值**就回落结构比对 —— 这比"一侧未知就判不等"更完整，
        # 且不会引入误报：结构相同才是同一份符号（`_M` vs `MARGIN` 结构不同，仍不匹配）。
        return _norm(a) == _norm(b)


def _cls_env(tree):
    """类属性合并表（`self.M` / 裸名 `M` 都查它）。"""
    env = {}
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            env.update(_assign_env(cls))
    return env


def _enclosing(tree):
    """`{节点 id: 所属函数名}` —— 调用点白名单要按函数归属。"""
    out = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for nd in ast.walk(fn):
                out[id(nd)] = fn.name
    return out


def _find_margin_clamps(repo, rel):
    """AST 档：结构化找 margin clamp + **调用点**分析。返回 `[{file,line,tier,form,fn}]`。

    抓的形状（inline）：`max(lo, min(hi, v))`，`hi` 是 `X - lo`（两处 `lo` **同值**）。
    抓的形状（Cap-2 调用点）：调用共享 helper 时，某实参是 `X - M` 而另一实参同值于 `M`
      ⇒ **边界算式被搬到了调用点**（即使 `max/min` 在 helper 里）。
    """
    info = repo.files.get(rel)
    if not info:
        return []
    tree = info["tree"]
    cls_env = _cls_env(tree)
    enc = _enclosing(tree)
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        env = dict(_assign_env(fn))
        # ★ 函数内 import 也要认（很常见）—— 合并进 imps 供跨文件解析
        imps = dict(info.get("imports") or {})
        imps.update(repo.local_imports(fn, rel))
        # ---------- inline 档 ----------
        for node in ast.walk(fn):
            if not (_is_builtin(node, "max") and len(node.args) == 2):
                continue
            lo, inner = node.args
            if not (_is_builtin(inner, "min") and len(inner.args) == 2):
                continue
            hi_x = resolve_left(inner.args[0], env)       # 名字 → 赋值表达式（一层）
            if not (isinstance(hi_x, ast.BinOp) and isinstance(hi_x.op, ast.Sub)):
                continue
            if not repo.same_bound(hi_x.right, lo, rel, env, cls_env, imps):
                continue
            left_x = resolve_left(hi_x.left, env)
            left_val, _ = repo.value(left_x, rel, env, cls_env, 0, imps)
            if left_val is not None:
                tier = "fixed-bound"          # 上界建立在**常数**上 ⇒ 概率/数值保护
            elif isinstance(lo, (ast.Name, ast.Attribute)):
                tier = "variable"
            else:
                tier = "literal"
            out.append({"file": rel, "line": int(getattr(node, "lineno", 0)), "tier": tier,
                        "fn": enc.get(id(node), "?"),
                        "form": "max(%s, min(%s - %s, ...))"
                                % (_norm(lo)[:26], _norm(left_x)[:26], _norm(lo)[:26])})
        # ---------- Cap-2：边界算式**经参数**进共享 helper ----------
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            callee = (node.func.id if isinstance(node.func, ast.Name)
                      else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if callee is None or not _is_shared_clamp(callee, repo, rel, imps):
                continue
            args = [resolve_left(a, env) if isinstance(a, ast.Name) and a.id in env else a
                    for a in node.args]
            for i, a in enumerate(args):
                if not (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub)):
                    continue
                for j, b in enumerate(args):
                    if i == j:
                        continue
                    if repo.same_bound(a.right, b, rel, env, cls_env, imps):
                        out.append({"file": rel, "line": int(getattr(node, "lineno", 0)),
                                    "tier": "call-arithmetic",
                                    "fn": enc.get(id(node), "?"),
                                    "form": "%s(..., %s - %s, ...) ← 边界算式在**调用点**"
                                            % (callee, _norm(a.left)[:22], _norm(a.right)[:22])})
                        break
    return out


def resolve_left(node, env):
    """把 `hi` 这类名字换成它的赋值表达式（一层；用于取"上界建立在哪里"）。"""
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    return node


def _is_shared_clamp(callee, repo, rel, imps=None):
    """该被调者是不是"共享的钳制实现"（认名字、**认别名**、或认出仓库里的泛型 clamp）。

    ⚠ 必须**解开别名**：真实代码里是 `from .coord_contract import clamp_axis as _cc_clamp_axis`
    再调 `_cc_clamp_axis(...)` —— 只看被调者的字面名会认不出来（本轮实测，M19）。
    """
    imps = imps if imps is not None else ((repo.files.get(rel) or {}).get("imports") or {})
    if callee in CANONICAL_HELPERS:
        return True
    if callee in imps:                       # 别名 → 原名
        _tmod, attr = imps[callee]
        if attr in CANONICAL_HELPERS:
            return True
    for f in repo.files.values():
        fn = f["funcs"].get(callee)
        if fn is None:
            continue
        for r in ast.walk(fn):
            if (isinstance(r, ast.Return) and isinstance(r.value, ast.Call)
                    and _is_builtin(r.value, "max") and len(r.value.args) == 2
                    and _is_builtin(r.value.args[1], "min")):
                return True
    return False


def scan_clamp_copies(root, overrides=None):
    """扫 `root` 下 .py ⇒ `[{file, line, tier, form|text, fn}]`（**只报，不判**）。

    判死档：`literal` / `variable` / `call-arithmetic`；
    只报档：`fixed-bound`（概率/数值保护，同形不同义）、`suspect`（正则档供人看）。

    `overrides`: `{relpath: 源码}` —— 用给定源码替代磁盘那份。
    ⚠ 必需：守卫必须扫**它正在测的那份源码**，否则"内存里改了、磁盘没动"的变异看不见。
    """
    repo = _Repo(root, overrides)
    hits = []
    for rel, info in sorted(repo.files.items()):
        hits.extend(_find_margin_clamps(repo, rel))
        # ---- 正则档：只报"可疑"，并跳过注释行 / 字符串字面量行 / 已由 AST 判过的行
        ast_lines = {h["line"] for h in hits if h["file"] == rel}
        str_lines = set()
        for nd in ast.walk(info["tree"]):
            if isinstance(nd, ast.Constant) and isinstance(nd.value, str):
                for _ln in range(getattr(nd, "lineno", 1),
                                 getattr(nd, "end_lineno", getattr(nd, "lineno", 1)) + 1):
                    str_lines.add(_ln)
        for i, ln in enumerate(info["src"].splitlines(), 1):
            if ln.lstrip().startswith("#") or i in ast_lines or i in str_lines:
                continue
            if (SUSPECT_RE.search(ln) and "0.5" in ln
                    and any(w in ln for w in SUSPECT_WORDS)):
                hits.append({"file": rel, "line": i, "tier": "suspect", "fn": "?",
                             "text": ln.strip()[:120]})
    return hits
