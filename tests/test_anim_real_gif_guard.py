# -*- coding: utf-8 -*-
"""`lingshu.world.anim` 守卫 —— 真动画 API 的**回退即红**判据。

为什么有本文件（Lead 定位）：
    CI 的 `tools/coverage_map_lint.py` 报 `lingshu/world/anim.py` **零引用**
    （`new_zero_ref=[{file: 'lingshu/world/anim.py', lines: 442}]`，本地不报是因为
    不在 PR 里的别的文件注释里恰好有 "anim.py" 文本）。
    ⇒ 它暴露的是**真缺口**：主线③的"产出真动画"这条能力**一直没有守卫**——
      库在、成品在被人工验，但**没有任何断言守得住它**。
    本文件把 anim-seq 报告里那批读数（`n_frames` 回读 / 逐帧质心位移 / 胶片条
    与动画可区分 / RGBA 合成 / 对抗性输入 / 跨进程确定性）固化成可复红断言。

判据来源：`anim-seq` 的实测读数 + `to_gif` 的 `_verify_gif` 回读契约。
回退即红（实测）：把 `to_gif` 的 `loop` 透传删掉 ⇒ A2 红；把 RGBA 帧改成直接
    `convert("P")`（不做 `background` 合成）⇒ E1 红；把 `n_frames` 回读断言抽掉
    ⇒ A1 红（帧数不再被核）。
跑法：`python -m pytest tests/test_anim_real_gif_guard.py -q`
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import anim as A                                      # noqa: E402

PIL = pytest.importorskip("PIL", reason="本判据要读回 GIF 帧数与像素（Pillow）")
from PIL import Image                                                     # noqa: E402

#: 帧尺寸（小图够用：判据看的是"在动"与"帧数"，不是画质）
W, H = 96, 72
#: 一个**确定在动**的合成渲染器：第 i 帧画一个沿 x 移动的色块。
#: 用它而不是真模型，是为了让"逐帧位移 > 0"这条判据**不依赖资产是否在本机**。
BALL = (232, 132, 86)
GROUND = (34, 40, 58)


class _Mover:
    """`anim.frames` 要求的面是 `render(w, h, **kw)`；`mutate(i, r)` 在渲染前调用。"""

    def __init__(self, n):
        self.n = n
        self.i = 0

    def set_frame(self, i):
        self.i = max(0, min(int(i), self.n - 1))

    def render(self, w, h, **_kw):
        img = Image.new("RGB", (w, h), GROUND)
        px = img.load()
        #: 球心沿 x 从 8% 走到 92%，y 走一条抛物线 ⇒ 每一帧都在动
        t = self.i / max(1, self.n - 1)
        cx = int(w * (0.08 + 0.84 * t))
        cy = int(h * (0.55 - 0.35 * math.sin(math.pi * t)))
        r = max(3, int(min(w, h) * 0.06))
        for y in range(max(0, cy - r), min(h, cy + r + 1)):
            for x in range(max(0, cx - r), min(w, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    px[x, y] = BALL
        return img


def _centroid(img):
    """球的质心。判据 = **色相**（球是唯一暖色）——不要用"与地面色差"，
    那会把背景/地面算成前景而把位移稀释成 ~0（Lead 实测踩过：min 0.0 px 假象）。"""
    px = img.convert("RGB").load()
    sx = sy = n = 0
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b = px[x, y]
            if r > g > b and (r - b) > 80:
                sx += x
                sy += y
                n += 1
    return (sx / n, sy / n, n) if n else (float("nan"), float("nan"), 0)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _make(n=12):
    r = _Mover(n)
    return A.frames(r, n, mutate=lambda i, rr: rr.set_frame(i), size=(W, H))


# =========================================================== A 真动画本体
def test_a1_gif_回读帧数与尺寸与时长(tmp_path):
    """A1：`to_gif` 必须产出**回读得到正确帧数**的真动画，且返回体自证。"""
    n = 12
    frames = _make(n)
    out = tmp_path / "a.gif"
    rep = A.to_gif(frames, str(out), duration_ms=40, loop=0)
    assert os.path.isfile(out), "GIF 没落盘"
    im = Image.open(str(out))
    assert getattr(im, "n_frames", 1) == n, (
        "回读帧数 %r != 输入 %d ⇒ 产出的不是真动画（可能是单帧或胶片条）"
        % (getattr(im, "n_frames", 1), n))
    assert im.size == (W, H), im.size
    assert im.info.get("duration") == 40, im.info.get("duration")
    assert im.info.get("loop") == 0, im.info.get("loop")
    #: 返回体必须自证（否则调用方拿不到可核对的读数）
    assert rep, "to_gif 必须返回可核对的回执"


def test_a2_每帧都在动_逐帧质心位移恒正():
    """A2：**逐帧质心位移恒 > 0** —— 这是"证明在动"的核心判据。
    回退（把帧序打乱/复用同一帧）⇒ 出现 0 位移 ⇒ 本组红。"""
    frames = _make(12)
    cs = [_centroid(f) for f in frames]
    assert all(c[2] > 0 for c in cs), "有帧找不到球（判据失效）"
    deltas = [math.hypot(cs[i][0] - cs[i - 1][0], cs[i][1] - cs[i - 1][1])
              for i in range(1, len(cs))]
    assert min(deltas) > 0.0, "存在零位移帧（画面冻住了）⇒ 不是真动画：%r" % (deltas,)
    assert sum(deltas) > 5.0, "总位移过小，看不出运动：%r" % (sum(deltas),)


def test_a3_胶片条与动画必须可区分():
    """A3：`to_strip` 是**一张静图**，回读 `n_frames` 必须 == 1。
    ⇒ 判据能分开"动画"与"胶片条"（否则 A1 的帧数断言就失去判别力）。"""
    frames = _make(6)
    strip = A.to_strip(frames, cols=3, background=GROUND)
    assert getattr(strip, "n_frames", 1) == 1, (
        "胶片条竟被读成多帧 ⇒ 判据分不开动画与胶片条")


def test_a4_跨进程确定性_同输入两次逐字节相同(tmp_path):
    """A4：同一帧序列两次导出 ⇒ GIF **逐字节相同**（可复现，不受 hash 随机化影响）。"""
    frames = _make(8)
    g1, g2 = tmp_path / "1.gif", tmp_path / "2.gif"
    A.to_gif(frames, str(g1), duration_ms=30, loop=0)
    #: 另起进程再导一次（换 PYTHONHASHSEED），排除"同进程内巧合一致"
    env = dict(os.environ, PYTHONHASHSEED="12345")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lingshu.world import anim as A\n"
        "class M:\n"
        "    def __init__(s,n): s.n=n; s.i=0\n"
        "    def set_frame(s,i): s.i=i\n"
        "    def render(s,w,h,**k):\n"
        "        import PIL.Image as I\n"
        "        import math\n"
        "        im=I.new('RGB',(w,h),(34,40,58)); px=im.load()\n"
        "        t=s.i/max(1,s.n-1); cx=int(w*(0.08+0.84*t)); cy=int(h*(0.55-0.35*math.sin(math.pi*t)))\n"
        "        r=max(3,int(min(w,h)*0.06))\n"
        "        for y in range(max(0,cy-r),min(h,cy+r+1)):\n"
        "            for x in range(max(0,cx-r),min(w,cx+r+1)):\n"
        "                if (x-cx)**2+(y-cy)**2<=r*r: px[x,y]=(232,132,86)\n"
        "        return im\n"
        "m=M(8)\n"
        "fr=A.frames(m,8,mutate=lambda i,rr: rr.set_frame(i),size=(96,72))\n"
        "A.to_gif(fr, %r, duration_ms=30, loop=0)\n"
    ) % (REPO, str(g2))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert _sha(str(g1)) == _sha(str(g2)), (
        "两次导出字节不同 ⇒ 动画不可复现（容器里混入了时间戳等易变字段）")


# =========================================================== B 边界与对抗
def test_b1_rgba帧按背景合成_不静默丢alpha(tmp_path):
    """B1：带 alpha 的帧必须**按 background 合成**（`anim-seq` 修的 fail-open：
    旧实现 `convert("P")` 直接丢 alpha ⇒ 半透明像素变全黑/全白，静默出错）。"""
    frames = []
    for i in range(3):
        im = Image.new("RGBA", (W, H), GROUND + (255,))
        px = im.load()
        #: ★ 每帧必须**逐像素不同**：PIL 的 GifImagePlugin 对"与前帧相同"的帧
        #:   会无条件跳过（只剩 1 帧时回退单帧），而 `to_gif` 的写后回读自校验
        #:   会当场 `ValueError`。这条是 anim.py 的**正确行为**（拒绝静默丢帧）。
        x0 = W // 5 + i * 7
        for y in range(H // 3, 2 * H // 3):
            for x in range(x0, x0 + W // 4):
                px[x, y] = (232, 132, 86, 128)      # 半透明
        frames.append(im)
    rep = A.to_gif(frames, str(tmp_path / "alpha.gif"), duration_ms=30, loop=0)
    txt = repr(rep)
    assert "alpha" in txt.lower(), (
        "带 alpha 的帧没有在回执里自证合成 ⇒ RGBA fail-open 可能回归：%s" % txt[:300])


def test_b2_对抗性输入必须_raise_且不留坏文件(tmp_path):
    """B2：非法输入必须**显式报错**（不静默产出坏文件）——`anim-seq` 实测 8/8 raise。"""
    good = _make(3)
    cases = [
        ("空帧序列", lambda: A.to_gif([], str(tmp_path / "x1.gif"))),
        ("尺寸不一致", lambda: A.to_gif(good + [Image.new("RGB", (W + 1, H), GROUND)],
                                        str(tmp_path / "x2.gif"))),
        ("时长非法", lambda: A.to_gif(good, str(tmp_path / "x3.gif"), duration_ms=-5)),
        ("loop 非法", lambda: A.to_gif(good, str(tmp_path / "x4.gif"), loop=-1)),
        ("cols 非法", lambda: A.to_strip(good, cols=0)),
    ]
    for name, fn in cases:
        with pytest.raises(Exception):
            fn()
    for bad in ("x1.gif", "x2.gif", "x3.gif", "x4.gif"):
        assert not os.path.isfile(str(tmp_path / bad)), (
            "%s 留下了坏文件 ⇒ 报错路径不干净" % bad)


def test_b3_零依赖_anim不引numpy():
    """B3：`anim.py` 的世界层**零 numpy**（D-005）；PIL 只允许函数内惰性 import。"""
    src = open(os.path.join(REPO, "lingshu", "world", "anim.py"), encoding="utf-8").read()
    assert "numpy" not in src, "world 层引了 numpy（违反 D-005）"
    #: 顶层的 `import PIL` 是硬依赖 ⇒ 不允许（必须在函数内惰性）
    top = [ln for ln in src.splitlines()
           if ln.startswith(("import ", "from ")) and "PIL" in ln]
    assert not top, "anim.py 顶层硬 import PIL ⇒ 违反惰性依赖纪律：%r" % top
