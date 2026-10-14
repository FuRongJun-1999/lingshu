# -*- coding: utf-8 -*-
"""rust_bridge · hex_nn Rust 后端的 Python 桥(R1)
====================================================================================
设计:训练计划(plan)与执行(execution)分离——
  Python 生成 plan(每步 batch 索引+支路采样索引,权威随机性),
  Rust hex_nn 是确定性执行器;验收=同 plan 下 D 曲线逐点对照(1e-9)。
Python 参考实现保留(hex_train/hex_hier),Rust 只做热区。
"""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _locate_hex_exe() -> Path:
    """hex_nn.exe 探测链(不硬编码本机路径纪律):
    环境变量 AEIS_HEX_NN → GitHub 恢复克隆位置 → 旧机位置。
    (2026-09-24 故障迁移:<local-path> Files\\2_ai 已格式化;protocol-compiler
     远端=github.com/FuRongJun-1999/ChineseProgramming,克隆于 <local-path>)"""
    candidates = [
        os.environ.get("AEIS_HEX_NN"),
        r"<local-path>",
        r"<local-path> Files\2_ai\protocol-compiler\hex_nn\target\release\hex_nn.exe",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return Path(candidates[-1])      # 保留旧缺省(run_rust_train 报 FileNotFoundError)


HEX_EXE = _locate_hex_exe()

MAGIC_DATA = 0x4845584441544131
MAGIC_PLAN = 0x4845584E4E525531


# ==================== plan 生成(权威随机性在 Python) ====================

def make_plan(n: int, steps: int, batch: int, samples: int,
              seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    plan_batch = [rng.permutation(n)[:batch].tolist() for _ in range(steps)]
    n_params = 6 * 7 + 6 * 6 * 7 + 9 * (6 + 3)          # K1=6, K2=6, head
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


def unpack_out(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    raw = path.read_bytes()
    assert int.from_bytes(raw[:8], "little") == 0x4845584E4E4F5554
    steps = int.from_bytes(raw[8:16], "little")
    vals = np.frombuffer(raw[16:], dtype=np.float64)
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
    threads: train_par 的 rayon 线程数(经环境变量)。"""
    wd = workdir or Path("data/rust_bridge")
    wd.mkdir(parents=True, exist_ok=True)
    pack_data(lat, obj_idx, shape_idx, wd / "data.bin")
    pack_plan(plan, wd / "plan.bin")
    pack_vec(vec0, k1, k2, wd / "vec.bin", k3=k3)
    env = None
    if threads or lr != 0.1:
        import os
        env = dict(os.environ)
        if threads:
            env["RAYON_NUM_THREADS"] = str(threads)
        env["HEX_LR"] = str(lr)
    subprocess.run([str(HEX_EXE), mode, str(wd / "data.bin"),
                    str(wd / "plan.bin"), str(wd / "vec.bin"),
                    str(wd / "out.bin")], check=True, env=env)
    return unpack_out(wd / "out.bin")


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
    plan = make_plan(len(lat), steps, batch, samples, seed)
    curve_py = py_executor_par(lat, obj_idx, shape_idx, vec0.copy(), plan, net)
    curve_rs, _, _ = run_rust_train(lat, obj_idx, shape_idx, vec0.copy(),
                                    plan, mode="train_par", threads=1)
    diff = float(np.abs(curve_py - curve_rs).max())
    return {"py_last": float(curve_py[-1]), "rust_last": float(curve_rs[-1]),
            "max_abs_diff": diff, "pass": bool(diff < tol), "steps": steps}


def determinism_check(lat: np.ndarray, obj_idx: np.ndarray,
                      shape_idx: np.ndarray, vec0: np.ndarray,
                      steps: int = 10, batch: int = 32, samples: int = 24,
                      seed: int = 7, threads_list=(1, 2, 4)) -> Dict:
    """多线程确定性:同一 plan 不同线程数输出必须逐位一致。"""
    plan = make_plan(len(lat), steps, batch, samples, seed)
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
                  threads_list=(1, 2, 4, 8)) -> Dict:
    """加速曲线:同 plan 不同线程数的每步耗时。"""
    plan = make_plan(len(lat), steps, batch, samples, seed)
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
    plan = make_plan(len(lat), steps, batch, samples, seed)
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
               seed: int = 11) -> Dict:
    plan = make_plan(len(lat), steps, batch, samples, seed)
    t0 = time.time()
    curve, vec, secs = run_rust_train(lat, obj_idx, shape_idx, vec0, plan)
    return {"steps": steps, "wall_seconds": round(time.time() - t0, 2),
            "engine_seconds": round(secs, 2),
            "d_first": round(float(curve[0]), 5),
            "d_last": round(float(curve[-1]), 5)}
