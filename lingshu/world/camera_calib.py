# -*- coding: utf-8 -*-
"""camera_calib · 相机规格 → 内参（真实相机进世界模型的第一个接口）· issue #1194
============================================================================
能力缺口（issue #1194）：`lingshu/world/` 43 个模块**无一做相机内参与标定**，
所有用到相机的地方相机都是**白给**的（`multiview.ViewObs.camera`、
`scene_model`/`scene_simulator` 由 `look_at` 构造、`world3d.Camera3D` 的
`fov` 是构造参数而非估计量）。因此「真实图像 → 世界模型」这条路上，
**第一个需要估计的量（内参）没有落点**。

为什么不能靠几何反解（负结果，见 issue #1194 正文）：两视图自标定在数学上
不可辨识——试过的 5 种表述（`det(E)=0`、主点平移、`σ₁²−σ₂²`、`w=1/f²`、
尺度无关 `tr(M²)=1/2`）判据在真值处精确为 0 但**一片平坦**。
⇒ 内参只能来自**相机规格**或**标定板**，本模块做前者。

三重冗余校验（本模块的可验证性来源）：
同一个相机有三条互不相干的**真实**表达：

    ① 物理：焦距 f(mm) + 传感器有效区宽 w(mm)
    ② 等效：35mm 等效焦距 f35（厂商常公布）
    ③ 视场：水平 FOV（厂商也常公布）

三者满足恒等关系：

    crop = 43.2666 / hypot(w, h)      # 43.2666 = 35mm 画幅对角线（定义值）
    f35  = f_mm × crop
    FOV  = 2·atan(w / (2·f_mm))
    fx   = f_mm / w × res_x

任给其中两个可解第三个。**若三个都来自厂商真实规格且互相吻合，这个换算就被
真实世界数据验证过了**，而不是自己跟自己对。

诚实边界：
  · 主点取画幅中心是**显式假设**（规格数据里没有主点），不是观测；
  · 只覆盖**针孔 + 径向畸变**能描述的画幅；鱼眼/超广角未覆盖；
  · 传感器尺寸取"有效成像区"规格值。厂商常给 `1/1.28"` 这类**光学格式名**，
    需查表换算，本表用行业标准的有效区域尺寸，不是格式名直接除；
  · 多摄模组的"等效焦距"常按 4:3 算，16:9 裁切后 FOV 会变——本模块不处理裁切；
  · 本模块**不估计**任何量，只做换算与自洽校验。真正的标定（棋盘格 + 畸变）
    需要线性代数，依赖口径待 issue #1194 裁定。

判据来源：issue #1194 提议第一步；画幅定义（35mm = 36×24mm）是定义值非实测值。
纯标准库 · 零外部依赖（沿用 `world/` 现有纪律 D-005）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


#: 35mm 全画幅对角线（mm）—— 定义值，不是实测值。
FULLFRAME_DIAG_MM = math.hypot(36.0, 24.0)          # 43.266615305567875


#: 真实传感器**有效成像区**尺寸（mm）。来源：行业标准画幅规格（非格式名换算）。
#: 键 → (宽, 高, 说明)
SENSORS: Dict[str, Tuple[float, float, str]] = {
    "fullframe":   (36.00, 24.00, "35mm 全画幅"),
    "aps-c_canon": (22.30, 14.90, "佳能 APS-C"),
    "aps-c":       (23.50, 15.60, "索尼/尼康/宾得/富士 APS-C"),
    "mft":         (17.30, 13.00, "M4/3"),
    "one_inch":    (13.20,  8.80, "1 英寸"),
    "1_1.7":       ( 7.60,  5.70, "1/1.7 英寸"),
    "1_2.3":       ( 6.17,  4.55, "1/2.3 英寸"),
}


class SpecError(ValueError):
    """相机规格不自洽（缺字段 / 数值非法 / 冗余校验不通过）。"""


def sensor_dims(name: str) -> Tuple[float, float]:
    """画幅名 → (宽 mm, 高 mm)。未知画幅**抛出**并列出已知项（不猜）。"""
    key = name.strip().lower()
    if key not in SENSORS:
        raise SpecError("sensor_dims: 未知画幅 %r；已知: %s"
                        % (name, ", ".join(sorted(SENSORS))))
    w, h, _ = SENSORS[key]
    return w, h


def crop_factor(sensor_width_mm: float, sensor_height_mm: float) -> float:
    """裁剪系数 = 全画幅对角线 / 传感器对角线。全画幅恰好 1.0（可作自检）。"""
    if sensor_width_mm <= 0 or sensor_height_mm <= 0:
        raise SpecError("crop_factor: 传感器尺寸必须为正")
    return FULLFRAME_DIAG_MM / math.hypot(sensor_width_mm, sensor_height_mm)


def focal35(lens_mm: float, sensor_width_mm: float,
            sensor_height_mm: float) -> float:
    """35mm 等效焦距。"""
    if lens_mm <= 0:
        raise SpecError("focal35: 焦距必须为正")
    return lens_mm * crop_factor(sensor_width_mm, sensor_height_mm)


def hfov_deg(lens_mm: float, sensor_width_mm: float) -> float:
    """水平视场角（度）。"""
    if lens_mm <= 0 or sensor_width_mm <= 0:
        raise SpecError("hfov_deg: 焦距与传感器宽度必须为正")
    return 2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * lens_mm)))


def vfov_deg(lens_mm: float, sensor_height_mm: float) -> float:
    """垂直视场角（度）。"""
    if lens_mm <= 0 or sensor_height_mm <= 0:
        raise SpecError("vfov_deg: 焦距与传感器高度必须为正")
    return 2.0 * math.degrees(math.atan(sensor_height_mm / (2.0 * lens_mm)))


def lens_from_focal35(f35: float, sensor_width_mm: float,
                      sensor_height_mm: float) -> float:
    """35mm 等效焦距 → 实际焦距(mm)（反解，用于与厂商数据对照）。"""
    if f35 <= 0:
        raise SpecError("lens_from_focal35: 等效焦距必须为正")
    return f35 / crop_factor(sensor_width_mm, sensor_height_mm)


def lens_from_hfov(hfov: float, sensor_width_mm: float) -> float:
    """水平视场角 → 焦距(mm)（反解）。"""
    if not (0.0 < hfov < 180.0):
        raise SpecError("lens_from_hfov: FOV 必须在 (0,180) 度，得到 %r" % (hfov,))
    if sensor_width_mm <= 0:
        raise SpecError("lens_from_hfov: 传感器宽度必须为正")
    return sensor_width_mm / (2.0 * math.tan(math.radians(hfov) / 2.0))


@dataclass
class CameraIntrinsics:
    """针孔内参 + 它来自哪条规格（全部字段可 JSON 往返，无隐藏状态）。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    sensor_w_mm: float
    sensor_h_mm: float
    lens_mm: float
    sensor: str = "unknown"
    principal_point_source: str = "center_assumption"

    # ---- 派生量（只读，由已有字段算出，不另存状态） ----
    @property
    def crop_factor(self) -> float:
        return crop_factor(self.sensor_w_mm, self.sensor_h_mm)

    @property
    def focal35_mm(self) -> float:
        return focal35(self.lens_mm, self.sensor_w_mm, self.sensor_h_mm)

    @property
    def hfov_deg(self) -> float:
        return hfov_deg(self.lens_mm, self.sensor_w_mm)

    @property
    def vfov_deg(self) -> float:
        return vfov_deg(self.lens_mm, self.sensor_h_mm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "camera_intrinsics/1",
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": int(self.width), "height": int(self.height),
            "sensor_w_mm": self.sensor_w_mm, "sensor_h_mm": self.sensor_h_mm,
            "lens_mm": self.lens_mm, "sensor": self.sensor,
            "principal_point_source": self.principal_point_source,
        }

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "CameraIntrinsics":
        need = ("fx", "fy", "cx", "cy", "width", "height",
                "sensor_w_mm", "sensor_h_mm", "lens_mm")
        miss = [k for k in need if k not in doc]
        if miss:
            raise SpecError("CameraIntrinsics.from_dict: 缺字段 %s" % (", ".join(miss),))
        return cls(
            fx=float(doc["fx"]), fy=float(doc["fy"]),
            cx=float(doc["cx"]), cy=float(doc["cy"]),
            width=int(doc["width"]), height=int(doc["height"]),
            sensor_w_mm=float(doc["sensor_w_mm"]),
            sensor_h_mm=float(doc["sensor_h_mm"]),
            lens_mm=float(doc["lens_mm"]),
            sensor=str(doc.get("sensor", "unknown")),
            principal_point_source=str(
                doc.get("principal_point_source", "center_assumption")),
        )


def intrinsics_from_spec(*, sensor: str, lens_mm: float,
                         resolution: Tuple[int, int],
                         principal_point: Optional[Tuple[float, float]] = None
                         ) -> CameraIntrinsics:
    """相机规格 → 针孔内参。

    像素焦距 `fx = fy = lens_mm / sensor_w_mm × res_x`（正方形像素假设，
    与"有效区按宽度归一"的行业惯例同式）。

    `principal_point` 缺省取画幅中心——这是**显式假设**，会写进
    `principal_point_source`，不让调用方误以为它是观测量。
    """
    w_mm, h_mm = sensor_dims(sensor)
    res_x, res_y = int(resolution[0]), int(resolution[1])
    if res_x <= 0 or res_y <= 0:
        raise SpecError("intrinsics_from_spec: 分辨率必须为正")
    if lens_mm <= 0:
        raise SpecError("intrinsics_from_spec: 焦距必须为正")
    fpx = lens_mm / w_mm * res_x
    if principal_point is None:
        cx, cy = res_x / 2.0, res_y / 2.0
        src = "center_assumption"
    else:
        cx, cy = float(principal_point[0]), float(principal_point[1])
        src = "given"
    return CameraIntrinsics(fx=fpx, fy=fpx, cx=cx, cy=cy,
                            width=res_x, height=res_y,
                            sensor_w_mm=w_mm, sensor_h_mm=h_mm,
                            lens_mm=lens_mm, sensor=sensor.strip().lower(),
                            principal_point_source=src)


def cross_check(*, sensor: str, lens_mm: float, resolution: Tuple[int, int],
                claim_focal35_mm: Optional[float] = None,
                claim_hfov_deg: Optional[float] = None,
                tol_frac: float = 0.02) -> Dict[str, Any]:
    """用厂商公布的 f35 / FOV 校验换算链是否自洽（本模块的核心验证入口）。

    `tol_frac` 默认 2%：厂商数据常四舍五入（"24mm 等效"实际 23.5mm），
    2% 是这类标称值本身的精度量级，不是放水。

    两个 claim 都缺时**抛出**——没有独立口径就没有校验，不能返回"通过"。
    """
    w_mm, h_mm = sensor_dims(sensor)
    if claim_focal35_mm is None and claim_hfov_deg is None:
        raise SpecError("cross_check: 至少给 claim_focal35_mm 或 claim_hfov_deg 之一"
                        "（没有独立口径就无法校验）")
    derived_f35 = focal35(lens_mm, w_mm, h_mm)
    derived_fov = hfov_deg(lens_mm, w_mm)
    out: Dict[str, Any] = {
        "sensor": sensor.strip().lower(), "sensor_mm": [w_mm, h_mm],
        "lens_mm": lens_mm,
        "derived_focal35_mm": round(derived_f35, 4),
        "derived_hfov_deg": round(derived_fov, 4),
        "crop_factor": round(crop_factor(w_mm, h_mm), 4),
        "tol_frac": tol_frac, "checks": [],
    }
    ok = True
    if claim_focal35_mm is not None:
        rel = abs(derived_f35 - claim_focal35_mm) / claim_focal35_mm
        passed = rel <= tol_frac
        ok = ok and passed
        back = lens_from_focal35(claim_focal35_mm, w_mm, h_mm)
        out["checks"].append({
            "what": "focal35", "claimed": claim_focal35_mm,
            "derived": round(derived_f35, 4), "rel_err": round(rel, 6),
            "pass": passed,
            "back_solved_lens_mm": round(back, 6),
            "back_solve_rel_err": round(abs(back - lens_mm) / lens_mm, 6),
        })
    if claim_hfov_deg is not None:
        rel = abs(derived_fov - claim_hfov_deg) / claim_hfov_deg
        passed = rel <= tol_frac
        ok = ok and passed
        back = lens_from_hfov(claim_hfov_deg, w_mm)
        out["checks"].append({
            "what": "hfov", "claimed": claim_hfov_deg,
            "derived": round(derived_fov, 4), "rel_err": round(rel, 6),
            "pass": passed,
            "back_solved_lens_mm": round(back, 6),
            "back_solve_rel_err": round(abs(back - lens_mm) / lens_mm, 6),
        })
    out["pass"] = ok
    out["intrinsics"] = intrinsics_from_spec(
        sensor=sensor, lens_mm=lens_mm, resolution=resolution).to_dict()
    return out


def self_check_fullframe() -> Dict[str, Any]:
    """全画幅自检：crop 必须**恰好** 1.0，50mm 的 f35 必须**恰好** 50。

    这不是"相信实现"，而是用**画幅定义**检验公式：裁剪系数对全画幅恒等于 1。
    若这条不成立，说明公式里的参考对角线或归一化写错了（缺陷即红）。
    """
    w, h = sensor_dims("fullframe")
    cf = crop_factor(w, h)
    f35 = focal35(50.0, w, h)
    return {"crop_factor": round(cf, 12), "focal35_of_50mm": round(f35, 12),
            "pass": abs(cf - 1.0) < 1e-12 and abs(f35 - 50.0) < 1e-12}
