# -*- coding: utf-8 -*-
"""rust_bridge · hex_nn Rust 后端的 Python 桥(R1)
====================================================================================
设计:训练计划(plan)与执行(execution)分离——
  Python 生成 plan(每步 batch 索引+支路采样索引,权威随机性),
  Rust hex_nn 是确定性执行器;验收=同 plan 下 D 曲线逐点对照(1e-9)。
Python 参考实现保留(hex_train/hex_hier),Rust 只做热区。
"""
from __future__ import annotations
from contextlib import nullcontext
import os
import subprocess
from tempfile import TemporaryDirectory
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

#: 包根（`<repo>/lingshu/nn/rust_bridge.py` → `<repo>`）。缺省 hex_nn.exe
#: 锚定于此，**不**按调用时 cwd 解析（#488）。
_PKG_ROOT = Path(__file__).resolve().parents[2]

#: 子进程 env **白名单·具名**（#488，兼修 #350）：跨平台「进程运行基座」——
#: 路径 / 临时目录 / 区域 / 主机自省。**不含任何凭据名**（取白名单而非黑名单的
#: 理由与 `#227` 同款：黑名单对凭据面 fail-open）。
_ENV_ALLOW_NAMES = frozenset({
    # 可执行查找 / 命令解释器
    "PATH", "PATHEXT", "COMSPEC",
    # Windows 目录基座（PE 加载器与 CRT 启动期读用）
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
    "PROGRAMDATA", "ALLUSERSPROFILE", "PUBLIC",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMW6432",
    # 临时目录
    "TEMP", "TMP", "TMPDIR",
    # 家目录 / 账户
    "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA",
    "USER", "USERNAME", "USERDOMAIN", "LOGNAME",
    # 区域 / 终端
    "LANG", "LANGUAGE", "TZ", "TERM",
    # 主机自省
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS",
    # 区域/编码面（#495，复核 #680 §一.3 残留）：原 `("LC_",)` 宽前缀顺带放行
    # 任意 `LC_*` 名，含凭据形状的 `LC_TOKEN` / `LC_PASSWORD` ⇒ 与
    # `brain_store._ENV_ALLOW_NAMES`（#227 续 / #652③）同款，收敛为 POSIX +
    # glibc 的 13 个区域名具名。非标准 `LC_*` 名本无区域语义。
    "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MESSAGES", "LC_MONETARY",
    "LC_NUMERIC", "LC_TIME",
    "LC_PAPER", "LC_NAME", "LC_ADDRESS", "LC_TELEPHONE", "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
})

#: 子进程 env **白名单·前缀**：**为空**（#495）——原仅余 `LC_`，该前缀会顺带
#: 放行 `LC_TOKEN` / `LC_PASSWORD` 这类凭据形状名，13 个区域名已改具名（见上）。
_ENV_ALLOW_PREFIXES = ()


def _env_is_allowed(name: str) -> bool:
    """继承的环境变量名是否在白名单内（大小写无关）。"""
    up = str(name).upper()
    if up in _ENV_ALLOW_NAMES:
        return True
    return any(up.startswith(p) for p in _ENV_ALLOW_PREFIXES)


class HexExeError(RuntimeError):
    """hex_nn.exe 不可用/不可信（#488）：fail-closed，不静默回退。"""


def _default_hex_exe() -> Path:
    """包根锚定的缺省 hex_nn.exe（**绝对**路径；绝不按调用时 cwd 解析）。"""
    return _PKG_ROOT / "hex_nn" / "target" / "release" / "hex_nn.exe"


def _configured_hex_exe_candidates() -> List[str]:
    """显式配置面（环境变量驱动，保持 #10 既有行为）：
    `AEIS_HEX_NN` → `AEIS_HEX_NN_DIR/hex_nn.exe`，按优先级排列。"""
    out: List[str] = []
    direct = os.environ.get("AEIS_HEX_NN")
    if direct:
        out.append(direct)
    d = os.environ.get("AEIS_HEX_NN_DIR")
    if d:
        out.append(os.path.join(d, "hex_nn.exe"))
    return out


def _locate_hex_exe() -> Path:
    """hex_nn.exe 探测链(不硬编码本机路径纪律):
    环境变量 AEIS_HEX_NN → 部署方声明目录 AEIS_HEX_NN_DIR 下的 hex_nn.exe
    → 缺省**包根锚定**的绝对路径。
    (2026-09-24 故障迁移:旧机 2_ai 工作区已格式化;protocol-compiler
     远端=github.com/FuRongJun-1999/ChineseProgramming)

    #10 修：导出脱敏把两个**本机字面量候选路径**改写成了中性占位符
    （`<local-path>`），而它们是**路径操作数**（进 `Path(c).is_file()` 探测链）
    ⇒ 脱敏后两候选恒 `is_file() == False`，探测链静默失效（候选里还留着
    一个永远不成立的假路径）。判据来源：同目录 `docs/intake/body-export-v0.1/
    README.md` §三.③ 的导出精修先例「本机字面量 → 环境变量（缺省 …）」
    （`activation.py` 的 `LINGSHU_DATA_ROOT`）。故本处同样改为环境变量驱动，
    缺省取**仓内相对**路径（不再是任何机器绝对路径 / 占位符）。

    #488 修：上面「缺省取仓内相对路径」正是漏洞本身——相对路径按**调用时
    cwd** 解析，不可信目录下的同名文件会被 subprocess 直接执行；且该值在
    import 时固定。现改为**包根锚定**的绝对路径，并在 `resolve_hex_exe()`
    里于**调用时**做 `is_absolute()` + `is_file()` 校验、fail-closed。
    本函数保留为 import 时的**宽松**快照（`HEX_EXE`），不做 fail-closed。
    """
    for c in _configured_hex_exe_candidates():
        if Path(c).is_file():
            return Path(c)
    return _default_hex_exe()


def resolve_hex_exe() -> Path:
    """**调用时**解析 hex_nn.exe 并校验（#488）：
    ①显式配置（AEIS_HEX_NN / AEIS_HEX_NN_DIR）优先——配置了但不可用则
      fail-closed（抛 `HexExeError`，不回退缺省，避免配置错误被静默吞掉）；
    ②未配置 → 包根锚定的绝对缺省；
    ③任何候选都要求 `is_absolute()` 且 `is_file()`，否则抛 `HexExeError`。
    绝不按调用时 cwd 解析——cwd 下的同名文件不得被当作 hex_nn.exe 执行。
    """
    configured = _configured_hex_exe_candidates()
    if configured:
        problems = []
        for c in configured:
            p = Path(c)
            if not p.is_absolute():
                problems.append("%s（非绝对路径）" % c)
                continue
            if p.is_file():
                return p
            problems.append("%s（文件不存在）" % c)
        raise HexExeError(
            "hex_nn.exe 显式配置不可用（#488 fail-closed，不回退缺省）："
            + "；".join(problems)
            + "。请把 AEIS_HEX_NN 设为存在的绝对路径，或清除该变量使用包根缺省。")
    default = _default_hex_exe()
    if not default.is_absolute():
        raise HexExeError("hex_nn.exe 缺省路径非绝对（#488）：%r" % str(default))
    if not default.is_file():
        raise HexExeError(
            "hex_nn.exe 缺省路径不存在（#488 fail-closed）：%s。"
            "请构建 hex_nn 或用 AEIS_HEX_NN 指向绝对路径。" % default)
    return default


def _minimal_subprocess_env(extra: Optional[Dict[str, str]] = None
                            ) -> Dict[str, str]:
    """子进程**最小环境**（#488，兼修 #350 的透传面）：**白名单放行**。

    取白名单而非「剔 `AEIS_*` + 凭据形状名」的黑名单，理由与 `#227` 同款：
    黑名单对凭据面天然 fail-open——新出现的凭据名只要不撞模式就照旧泄露
    （`IGCCSVC_DB` / `YOUR_STEP_API_KEY` 在 `#227` 已证）。白名单是
    「不声明就不给」：只透传进程运行基座（路径 / 临时目录 / 区域 / 主机自省），
    其余继承变量一律**不透传**。

    显式 `extra`（`RAYON_NUM_THREADS` / `HEX_LR`）在收窄**之后**叠加——那是
    调用点有意传的配置，属显式面。
    """
    env = {k: v for k, v in os.environ.items() if _env_is_allowed(k)}
    if extra:
        env.update(extra)
    return env


HEX_EXE = _locate_hex_exe()

MAGIC_DATA = 0x4845584441544131
MAGIC_PLAN = 0x4845584E4E525531


# ==================== plan 生成(权威随机性在 Python) ====================

def make_plan(n: int, steps: int, batch: int, samples: int,
              seed: int, k1: int = 6, k2: int = 6) -> Dict:
    """生成 plan。**#294**：支路采样上限随 `k1/k2` 走，不再写死 `375`。

    原实现把 `n_params` 硬编码成 `6*7 + 6*6*7 + 9*(6+3)`，只在 `K1=K2=6`
    时与真实参数量巧合相等；`K1/K2` 取其它值时 `plan_sample` 会给出越界下标，
    下游 `py_executor_par` 的 `v[pi]` 直接 `IndexError`（硬崩，不是静默）。

    口径与 `hex_hier.HexHierNet.n_params()` 对齐：
        n_params = K1*7 + K2*K1*7 + 9*(K2+3)
    其中 `9 = len(OBJ)`、`K2+3 = head_in`（`hex_hier.py` head_in = n_kernels2+3）。
    **注意**（已知边界，未在本件扩范围）：`stacked=False` 时 head_in 为 `K1+3`、
    `deep_norm and amp_sep` 时 head_in 额外 +1，本式不覆盖这两种配置。
    """
    rng = np.random.default_rng(seed)
    plan_batch = [rng.permutation(n)[:batch].tolist() for _ in range(steps)]
    n_params = k1 * 7 + k1 * k2 * 7 + 9 * (k2 + 3)      # 与 HexHierNet.n_params() 同口径
    plan_sample = [rng.choice(n_params, size=min(samples, n_params),
                              replace=False).tolist() for _ in range(steps)]
    return {"plan_batch": plan_batch, "plan_sample": plan_sample}


def pack_plan(plan: Dict, path: Path) -> Path:
    pb, ps = plan["plan_batch"], plan["plan_sample"]
    out = bytearray()
    out += np.uint64(MAGIC_PLAN).tobytes()
    out += np.uint64(len(pb)).tobytes()
    out += np.uint64(len(pb[0])).tobytes()
    out += np.uint64(len(ps[0])).tobytes()
    for v in pb:
        out += np.array(v, dtype=np.uint32).tobytes()
    for v in ps:
        out += np.array(v, dtype=np.uint32).tobytes()
    path.write_bytes(bytes(out))
    return path


# ==================== data / vec 打包 ====================

def pack_data(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
              path: Path) -> Path:
    n, r, c = lat.shape[0], lat.shape[1], lat.shape[2]
    out = bytearray()
    out += np.uint64(MAGIC_DATA).tobytes()
    out += np.uint64(n).tobytes()
    out += np.uint64(r).tobytes()
    out += np.uint64(c).tobytes()
    out += np.array(obj_idx, dtype=np.uint32).tobytes()
    out += np.array(shape_idx, dtype=np.uint32).tobytes()
    out += np.ascontiguousarray(lat, dtype=np.float64).tobytes()
    path.write_bytes(bytes(out))
    return path


def pack_vec(vec: np.ndarray, k1: int, k2: int, path: Path, k3: int = 0) -> Path:
    out = bytearray()
    out += np.uint64(k1).tobytes()
    out += np.uint64(k2).tobytes()
    if k3:
        out += np.uint64(k3).tobytes()
    out += np.ascontiguousarray(vec, dtype=np.float64).tobytes()
    path.write_bytes(bytes(out))
    return path


_MAGIC = 0x4845584E4E4F5554
_HEAD = 16  # 8 字节魔数 + 8 字节 steps


def unpack_out(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """读取 .hexout：魔数须为 HEXNNOUT，且长度自洽。

    ★ 原实现用 `assert` 校验魔数（#454）：**`python -O` 下 assert 整体消失**，
    非法二进制会被静默当成合法结果返回（实测 `-O` 时 trash 文件也返回三个值）。
    外部数据校验不得依赖 assert —— 改为显式 raise（与本模块 `HexExeError` 同口径）。
    另补长度自洽：载入期把「长度不足 / steps 越界」挡在门口，而不是留给下游
    在 `vals[:steps]` 处产出**空数组**这种更难定位的形态。
    """
    raw = path.read_bytes()
    if len(raw) < _HEAD + 8:
        raise HexExeError(
            "unpack_out：文件过短（%d 字节，至少需 %d）——非 .hexout 或已截断：%r"
            % (len(raw), _HEAD + 8, str(path)))
    magic = int.from_bytes(raw[:8], "little")
    if magic != _MAGIC:
        raise HexExeError(
            "unpack_out：魔数不符（期望 0x%X，实得 0x%X）——非 .hexout：%r"
            % (_MAGIC, magic, str(path)))
    steps = int.from_bytes(raw[8:16], "little")
    vals = np.frombuffer(raw[16:], dtype=np.float64)
    # 末尾 1 个 float 是标量，其余按 steps 切曲线
    if steps < 0 or steps > len(vals) - 1:
        raise HexExeError(
            "unpack_out：steps=%d 与载荷不符（载荷 %d 个 float64，最多允许 %d）"
            % (steps, len(vals), len(vals) - 1))
    curve = vals[:steps]
    vec = vals[steps:-1]
    return curve, vec, float(vals[-1])


def run_rust_train(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
                   vec0: np.ndarray, plan: Dict, k1: int = 6, k2: int = 6,
                   workdir: Optional[Path] = None,
                   mode: str = "train", threads: Optional[int] = None,
                   lr: float = 0.1, k3: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """打包→调 hex_nn→解包。返回 (D 曲线, 最终权重, 秒)。
    mode: train(串行坐标下降) | train_par(并行批量评估-收敛)。
    threads: train_par 的 rayon 线程数(经环境变量)。

    缺省工作目录由每次调用独占，返回或失败后清理，不写入调用方 cwd。
    显式 workdir 保留四个诊断文件；并发调用方须为各任务指定不同目录。
    """
    exe = resolve_hex_exe()          # #488：调用时解析 + fail-closed（先于任何落盘）
    # #392：四个固定文件名只在本调用独占的默认目录中共享；显式目录仍归调用方。
    scope = TemporaryDirectory(prefix="lingshu_hex_") if workdir is None else nullcontext(workdir)
    with scope as directory:
        wd = Path(directory)
        wd.mkdir(parents=True, exist_ok=True)
        pack_data(lat, obj_idx, shape_idx, wd / "data.bin")
        pack_plan(plan, wd / "plan.bin")
        pack_vec(vec0, k1, k2, wd / "vec.bin", k3=k3)
        extra: Dict[str, str] = {}
        if threads:
            extra["RAYON_NUM_THREADS"] = str(threads)
        if lr != 0.1:
            extra["HEX_LR"] = str(lr)
        env = _minimal_subprocess_env(extra)     # #488：子进程最小环境（不透传凭据）
        subprocess.run([str(exe), mode, str(wd / "data.bin"),
                        str(wd / "plan.bin"), str(wd / "vec.bin"),
                        str(wd / "out.bin")], check=True, env=env)
        curve, vector, seconds = unpack_out(wd / "out.bin")
        if len(curve) != len(plan["plan_batch"]) or vector.size != np.asarray(vec0).size:
            raise ValueError("hex_nn 输出的曲线/权重长度与本次输入不匹配")
        return curve, vector, seconds


# ==================== Python 参考执行器(同 plan 确定性) ====================

def py_executor(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
                vec0: np.ndarray, plan: Dict, net, lr: float = 0.1,
                eps: float = 1e-3, deadzone: float = 1e-6) -> np.ndarray:
    """按 plan 逐位复现 Rust 语义的 Python 执行器(验收基准)。
    net: HexHierNet(stacked),其 set_vec 提供参数视图。"""
    import json
    curve = []
    vec = vec0.copy()
    k1, k2 = net.K, net.K2
    for step in range(len(plan["plan_batch"])):
        bidx = plan["plan_batch"][step]
        xb = lat[bidx]
        ob = obj_idx[bidx]
        sb = shape_idx[bidx]

        def loss(v: np.ndarray) -> float:
            net.set_vec(v)
            sf, cf = net.l2_features(xb)
            logits = net.l3_logits(sf, cf)
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            d_obj = -np.log(p[np.arange(len(ob)), ob] + 1e-12).mean()
            p_shape = np.stack([p[:, i * 3:(i + 1) * 3].sum(axis=1)
                                for i in range(3)], axis=1)
            d_shape = -np.log(p_shape[np.arange(len(sb)), sb] + 1e-12).mean()
            return float(d_obj + 0.5 * d_shape)

        d = loss(vec)
        curve.append(d)
        for pi in plan["plan_sample"][step]:
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            dp = loss(vp); dm = loss(vm)
            g = (dp - dm) / (2 * eps)
            if abs(g) >= deadzone:
                vec[pi] -= lr * np.sign(g)
    net.set_vec(vec)
    return np.array(curve)


def py_executor_par(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
                    vec0: np.ndarray, plan: Dict, net, lr: float = 0.1,
                    eps: float = 1e-3, deadzone: float = 1e-6) -> np.ndarray:
    """批量评估-收敛语义的 Python 参考(与 Rust train_steps_par 逐位对照)。
    每支路基于步初快照独立评估,全部完成后统一收敛。"""
    curve = []
    vec = vec0.copy()
    for step in range(len(plan["plan_batch"])):
        bidx = plan["plan_batch"][step]
        xb = lat[bidx]
        ob = obj_idx[bidx]
        sb = shape_idx[bidx]

        def loss(v: np.ndarray) -> float:
            net.set_vec(v)
            sf, cf = net.l2_features(xb)
            logits = net.l3_logits(sf, cf)
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            d_obj = -np.log(p[np.arange(len(ob)), ob] + 1e-12).mean()
            p_shape = np.stack([p[:, i * 3:(i + 1) * 3].sum(axis=1)
                                for i in range(3)], axis=1)
            d_shape = -np.log(p_shape[np.arange(len(sb)), sb] + 1e-12).mean()
            return float(d_obj + 0.5 * d_shape)

        d = loss(vec)
        curve.append(d)
        v = vec.copy()                                  # 步初快照
        updates = []
        for pi in plan["plan_sample"][step]:
            vp, vm = v.copy(), v.copy()
            vp[pi] += eps; vm[pi] -= eps
            dp = loss(vp); dm = loss(vm)
            g = (dp - dm) / (2 * eps)
            updates.append((pi, abs(g) >= deadzone, -lr * float(np.sign(g))))
        for pi, ok, delta in updates:                   # 统一收敛
            if ok:
                vec[pi] += delta
    net.set_vec(vec)
    return np.array(curve)


# ==================== R2 · 并行等价/确定性/加速 ====================

def equivalence_par(net, lat: np.ndarray, obj_idx: np.ndarray,
                    shape_idx: np.ndarray, steps: int = 12, batch: int = 32,
                    samples: int = 24, seed: int = 7,
                    tol: float = 1e-9) -> Dict:
    """Python 批量参考 ↔ Rust train_par(单线程)逐位等价。"""
    vec0 = net.get_vec().copy()
    plan = make_plan(len(lat), steps, batch, samples, seed,
                     k1=net.K, k2=net.K2)
    curve_py = py_executor_par(lat, obj_idx, shape_idx, vec0.copy(), plan, net)
    curve_rs, _, _ = run_rust_train(lat, obj_idx, shape_idx, vec0.copy(),
                                    plan, mode="train_par", threads=1)
    diff = float(np.abs(curve_py - curve_rs).max())
    return {"py_last": float(curve_py[-1]), "rust_last": float(curve_rs[-1]),
            "max_abs_diff": diff, "pass": bool(diff < tol), "steps": steps}


def determinism_check(lat: np.ndarray, obj_idx: np.ndarray,
                      shape_idx: np.ndarray, vec0: np.ndarray,
                      steps: int = 10, batch: int = 32, samples: int = 24,
                      seed: int = 7, threads_list=(1, 2, 4),
                      k1: int = 6, k2: int = 6) -> Dict:
    """多线程确定性:同一 plan 不同线程数输出必须逐位一致。"""
    plan = make_plan(len(lat), steps, batch, samples, seed, k1=k1, k2=k2)
    ref = None
    out = {}
    for th in threads_list:
        curve, _, _ = run_rust_train(lat, obj_idx, shape_idx, vec0.copy(),
                                     plan, mode="train_par", threads=th)
        if ref is None:
            ref = curve
            out["ref_last"] = float(curve[-1])
        else:
            out[f"threads_{th}_match"] = bool(np.array_equal(curve, ref))
    return out


def bench_threads(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
                  vec0: np.ndarray, steps: int = 60, batch: int = 96,
                  samples: int = 64, seed: int = 11,
                  threads_list=(1, 2, 4, 8), k1: int = 6, k2: int = 6) -> Dict:
    """加速曲线:同 plan 不同线程数的每步耗时。"""
    plan = make_plan(len(lat), steps, batch, samples, seed, k1=k1, k2=k2)
    res = {}
    base = None
    for th in threads_list:
        t0 = time.time()
        curve, _, engine = run_rust_train(lat, obj_idx, shape_idx,
                                          vec0.copy(), plan,
                                          mode="train_par", threads=th)
        wall = time.time() - t0
        res[f"t{th}"] = {"wall_s": round(wall, 2),
                         "ms_per_step": round(wall / steps * 1000, 1),
                         "d_last": round(float(curve[-1]), 4)}
        if base is None:
            base = wall
        res[f"t{th}"]["speedup"] = round(base / max(1e-9, wall), 2)
    return res


# ==================== 双后端等价验收 ====================

def equivalence_check(net, lat: np.ndarray, obj_idx: np.ndarray,
                      shape_idx: np.ndarray, steps: int = 12, batch: int = 32,
                      samples: int = 24, seed: int = 7,
                      tol: float = 1e-9) -> Dict:
    """同 plan 下 Python vs Rust D 曲线逐点对照。"""
    vec0 = net.get_vec().copy()
    plan = make_plan(len(lat), steps, batch, samples, seed,
                     k1=net.K, k2=net.K2)
    curve_py = py_executor(lat, obj_idx, shape_idx, vec0, plan, net)
    curve_rs, vec_rs, secs = run_rust_train(lat, obj_idx, shape_idx,
                                            vec0, plan)
    diff = float(np.abs(curve_py - curve_rs).max())
    return {"py_last": float(curve_py[-1]), "rust_last": float(curve_rs[-1]),
            "max_abs_diff": diff, "pass": bool(diff < tol),
            "rust_seconds": secs, "steps": steps}


# ==================== 基准 ====================

def bench_rust(lat: np.ndarray, obj_idx: np.ndarray, shape_idx: np.ndarray,
               vec0: np.ndarray, steps: int, batch: int, samples: int,
               seed: int = 11, k1: int = 6, k2: int = 6) -> Dict:
    plan = make_plan(len(lat), steps, batch, samples, seed, k1=k1, k2=k2)
    t0 = time.time()
    curve, vec, secs = run_rust_train(lat, obj_idx, shape_idx, vec0, plan)
    return {"steps": steps, "wall_seconds": round(time.time() - t0, 2),
            "engine_seconds": round(secs, 2),
            "d_first": round(float(curve[0]), 5),
            "d_last": round(float(curve[-1]), 5)}
