# -*- coding: utf-8 -*-
"""守卫 · 零引用参数（死参数）批次修复——五处签名不再吃「 body 里从未读过的参数 」
====================================================================================
本文件只钉「死参数不再存在」，不是「代码能跑」（能跑由各模块既有测试负责）。

缺陷面（五处，均为「形参在函数体内零引用」——调用方郑重传参、函数体从不读）：

  1. HexNet.grow_kernel(seed)          lingshu/nn/hex_train.py
     调用方 train_infogap 曾传 seed=seed+step，函数体核选择按 K 轮转（手写核族），
     seed 全程未读。删参（核族轮转是 docstring 点名的设计，确定性优先）。
  2. four_state_classify(net, ...)     lingshu/nn/hex_train.py
     判定只用 logits+cards，net 零引用。删参。
  3. pattern_mask(shape, ...)          lingshu/gen/hexgen_self_source.py
     花纹周期与 rad 挂钩（docstring 自证），shape 零引用；调用方还写了无谓的
     shape 净化表达式。删参＋删净化。
  4. texture_features(img, sil, core)  lingshu/gen/hexgen_c1_real.py
     纹理读数只用 img+core+bg，sil 零引用。删参。
  5. World3D._draw_object(..., horizon_px)  lingshu/world/world3d.py
     地平线只用于铺地面矩形（调用方局部变量），物体绘制不读它。删参。

判据（每条都可独立判红）：
  · AST 检测器：对上述五个函数逐一断言「每个形参都在函数体内被引用 ≥1 次」
    ——谁把死参数加回来，本组断言即点名判红；
  · 历史死参名缺席：seed / net / shape / sil / horizon_px 不在对应签名里；
  · 定点变异自证：把五处修复逐处「还原成带死参的旧签名」（真实源码字符串手术），
    检测器必须逐条点名判红——证明判据不是空转；
  · 行为钉：grow_kernel 核族轮转＋mix 列置零＋同起点同生长；four_state_classify
    零输入 BLINDSPOT（不再需要 net）；pattern_mask solid 全零＋纯函数；
    texture_features 平涂核心区 tex_rel=0。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_param_wiring.py -q --no-header
"""
import ast
import inspect
import textwrap

import numpy as np

from lingshu.gen import hexgen_c1_real
from lingshu.gen import hexgen_self_source
from lingshu.nn import hex_train
from lingshu.world import world3d

# ==================== 检测器 ====================

def _dead_params_of_source(src):
    """解析源码，返回 {函数名: [零引用形参]}（self/cls 除外）。

    引用计为函数体语句里出现的 Name 节点（嵌套闭包捕获也算用——语义正确）；
    形参默认值与注解不在 body 内，不计。"""
    tree = ast.parse(textwrap.dedent(src))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            params = [a.arg for a in args if a.arg not in ("self", "cls")]
            body_tree = ast.Module(body=node.body, type_ignores=[])
            used = {n.id for n in ast.walk(body_tree) if isinstance(n, ast.Name)}
            out[node.name] = [p for p in params if p not in used]
    return out


def _dead_params_of(fn):
    return _dead_params_of_source(inspect.getsource(fn)).get(fn.__name__, [])


# 五处修复面（函数对象 → 历史死参名）
FIXED = [
    (hex_train.HexNet.grow_kernel, "seed"),
    (hex_train.four_state_classify, "net"),
    (hexgen_self_source.pattern_mask, "shape"),
    (hexgen_c1_real.texture_features, "sil"),
    (world3d.World3D._draw_object, "horizon_px"),
]

# ==================== 正向断言 ====================

def test_fixed_functions_have_no_dead_params():
    """五处修复面：每个形参都必须在函数体内被引用（死参数复发即点名判红）。"""
    for fn, _dead_name in FIXED:
        dead = _dead_params_of(fn)
        assert dead == [], (
            f"{fn.__qualname__} 死参数复发（body 零引用）: {dead}")


def test_historical_dead_param_names_absent_from_signatures():
    """历史死参名不得再出现在对应签名里（抽掉修复 ⇒ 本组逐条判红）。"""
    for fn, dead_name in FIXED:
        sig = inspect.signature(fn)
        assert dead_name not in sig.parameters, (
            f"{fn.__qualname__} 签名重新吃回历史死参 {dead_name!r}: {sig}")


# ==================== 定点变异自证（抽掉修复 ⇒ 必红并点名） ====================

def test_detector_flags_synthetic_dead_param():
    """检测器自身先过秤：干净函数零报；注入死参必须被点名（判据不空转的最小证明）。"""
    clean = _dead_params_of_source(
        "def f(a, b):\n    return a + b\n")
    assert clean == {"f": []}, clean
    dirty = _dead_params_of_source(
        "def f(a, b, dead_one):\n    return a + b\n")
    assert dirty == {"f": ["dead_one"]}, dirty


def test_mutation_reverting_each_fix_is_flagged_by_name():
    """对真实源码做字符串手术，把五处修复逐处还原成带死参的旧签名——
    检测器必须逐条点名（多红＝纠缠，漏红＝判据空转）。"""
    # (函数, 修复后签名片段, 带死参的旧签名片段, 死参名)
    reverts = [
        (hex_train.HexNet.grow_kernel,
         "def grow_kernel(self):",
         "def grow_kernel(self, seed: int = 7):",
         "seed"),
        (hex_train.four_state_classify,
         "def four_state_classify(logits: np.ndarray, cards: List[Dict]) -> List[Dict]:",
         "def four_state_classify(net: HexNet, logits: np.ndarray, cards: List[Dict]) -> List[Dict]:",
         "net"),
        (hexgen_self_source.pattern_mask,
         "def pattern_mask(size, cx, cy, rad, pattern, ss=SS):",
         "def pattern_mask(shape, size, cx, cy, rad, pattern, ss=SS):",
         "shape"),
        (hexgen_c1_real.texture_features,
         "def texture_features(img, core, bg=None):",
         "def texture_features(img, sil, core, bg=None):",
         "sil"),
        (world3d.World3D._draw_object,
         "screen_w: int, screen_h: int) -> None:",
         "screen_w: int, screen_h: int, horizon_px: float) -> None:",
         "horizon_px"),
    ]
    for fn, fixed_sig, dead_sig, dead_name in reverts:
        src = inspect.getsource(fn)
        assert fixed_sig in src, (
            f"{fn.__qualname__} 签名与修复版不符（变异锚点漂移）: {src.splitlines()[0]}")
        mutated = _dead_params_of_source(src.replace(fixed_sig, dead_sig))
        assert mutated.get(fn.__name__) == [dead_name], (
            f"还原死参 {dead_name!r} 后检测器未逐条点名: {mutated}")


# ==================== 行为钉（修复面的语义不被删参破坏） ====================

def test_grow_kernel_cycles_family_deterministically():
    """grow_kernel：+1 核按 K 轮转取手写核族、mix 对应列置零、同起点同生长。"""
    names = list(hex_train.DEFAULT_KERNELS)
    net = hex_train.HexNet(n_class=2, seed=7)
    K0 = net.K
    net.grow_kernel()
    assert net.K == K0 + 1, f"生长后 K 应 +1: {net.K}"
    np.testing.assert_array_equal(
        net.conv[K0], hex_train.DEFAULT_KERNELS[names[K0 % len(names)]])
    assert not net.mix[:, K0:].any(), "生长核的 mix 列应为零（初始不影响输出）"
    net2 = hex_train.HexNet(n_class=2, seed=7)
    net2.grow_kernel()
    np.testing.assert_array_equal(net.conv[K0], net2.conv[K0])


def test_four_state_classify_zero_input_blindspot_without_net():
    """four_state_classify：零输入 → BLINDSPOT（判定只需 logits+cards，无 net）。"""
    logits = np.zeros((1, 4))
    cards = [{"class": c, "p_threshold": 0.5, "margin_threshold": 1.0}
             for c in range(4)]
    r = hex_train.four_state_classify(logits, cards)[0]
    assert r["verdict"] == "BLINDSPOT", f"零输入应诚实降级: {r}"


def test_pattern_mask_solid_zero_and_pure():
    """pattern_mask：solid 全零；同参同值（纯函数）；条纹确有调制。"""
    m = hexgen_self_source.pattern_mask(size=32, cx=16.0, cy=16.0, rad=8.0,
                                        pattern="solid")
    assert m.shape == (32, 32), m.shape
    assert not m.any(), "solid 应返回全 0（无调制）"
    a = hexgen_self_source.pattern_mask(size=32, cx=16.0, cy=16.0, rad=8.0,
                                        pattern="striped")
    b = hexgen_self_source.pattern_mask(size=32, cx=16.0, cy=16.0, rad=8.0,
                                        pattern="striped")
    np.testing.assert_array_equal(a, b)
    assert a.any(), "striped 应有非零调制"


def test_texture_features_flat_core_zero_tex():
    """texture_features：平涂核心区 tex_rel=0（读数只用 img+core+bg）。"""
    img = np.full((16, 16, 3), 120, np.uint8)
    core = np.zeros((16, 16), bool)
    core[4:12, 4:12] = True
    bg = np.array([10, 200, 10], np.uint8)
    tex = hexgen_c1_real.texture_features(img, core, bg=bg)
    assert tex["tex_rel"] == 0.0, f"平涂核心区 tex_rel 应为 0: {tex}"
