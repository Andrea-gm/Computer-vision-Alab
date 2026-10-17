"""Method H: B2-anchored detections refined with F geometry.

B2 remains the acceptance authority so F can never introduce a detection.
When an F box substantially overlaps a B2 box, F's tighter cap/body geometry
replaces the B2 geometry.  Unmatched B2 boxes are retained unchanged.
"""

from __future__ import annotations

try:
    from sideways_race.methods.method_b2_improved import (
        _iou,
        detect_sideways as _detect_b2,
    )
    from sideways_race.methods.method_f_cap_depth_hybrid import (
        detect_sideways as _detect_f,
    )
except ModuleNotFoundError:  # Direct ``python methods/method_h_b2f_merge.py``.
    from method_b2_improved import _iou, detect_sideways as _detect_b2
    from method_f_cap_depth_hybrid import detect_sideways as _detect_f

NAME = "H_b2f_merge"

_AGREEMENT_IOU = 0.55


def _merge_geometry(
    b2_boxes: list[tuple[int, int, int, int]],
    f_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Refine B2 boxes with one-to-one, IoU-agreeing F matches."""
    result = list(b2_boxes)
    candidates = sorted(
        (
            (_iou(b2_box, f_box), b2_index, f_index)
            for b2_index, b2_box in enumerate(b2_boxes)
            for f_index, f_box in enumerate(f_boxes)
        ),
        reverse=True,
    )
    used_b2: set[int] = set()
    used_f: set[int] = set()
    for overlap, b2_index, f_index in candidates:
        if overlap < _AGREEMENT_IOU:
            break
        if b2_index in used_b2 or f_index in used_f:
            continue
        result[b2_index] = f_boxes[f_index]
        used_b2.add(b2_index)
        used_f.add(f_index)
    return result


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Return B2 detections, using F only to refine agreed geometry."""
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []

    b2_boxes = [
        tuple(int(v) for v in box)
        for box in (_detect_b2(color_bgr, depth_mm, intrinsics) or [])
    ]
    if not b2_boxes:
        return []

    f_boxes = [
        tuple(int(v) for v in box)
        for box in (_detect_f(color_bgr, depth_mm, intrinsics) or [])
    ]
    return _merge_geometry(b2_boxes, f_boxes)


def _self_test() -> bool:
    b2 = [(10, 20, 80, 40), (150, 25, 70, 35)]
    f = [(8, 19, 82, 40), (260, 30, 60, 30)]
    merged = _merge_geometry(b2, f)
    print(f"NAME={NAME} b2={b2} f={f} merged={merged}")
    # The agreeing F box refines B2; the F-only box is never admitted.
    return merged == [f[0], b2[1]] and len(merged) == len(b2)


if __name__ == "__main__":
    if not _self_test():
        raise SystemExit(1)
    print("self-test OK")
