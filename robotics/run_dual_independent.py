"""Legacy dual vial launcher — DISABLED unless --enable-vial-detect.

Use rack tools instead:
    python rack_lid_state.py focus
    python rack_capture_agents.py run --camera 1 --seconds 30
"""

from __future__ import annotations

from depthcamera import RACK_TOOL_HINT, build_parser, launch_dual_independent


def main() -> None:
    args = build_parser().parse_args()
    args.dual_independent = True
    if not bool(getattr(args, "enable_vial_detect", False)):
        print(RACK_TOOL_HINT)
        raise SystemExit(2)
    raise SystemExit(launch_dual_independent(args))


if __name__ == "__main__":
    main()
