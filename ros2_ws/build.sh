#!/usr/bin/env bash
# Build orbbec_dual_sync with colcon. This is the supported multi-cam sync path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail() {
  echo ""
  echo "ERROR: $*" >&2
  echo ""
  echo "Supported path requires ROS2 + colcon:" >&2
  echo "  1) Install ROS2 Humble or Jazzy (Linux or WSL2 Ubuntu)" >&2
  echo "  2) source /opt/ros/\$ROS_DISTRO/setup.bash" >&2
  echo "  3) cd $ROOT && ./build.sh" >&2
  echo "  4) source install/setup.bash" >&2
  echo "  5) ros2 launch orbbec_dual_sync dual_sync_detect.launch.py" >&2
  echo ""
  echo "On Windows (native PowerShell, no bash): see ROS_SYNC.md" >&2
  echo "  .\\install_windows_ros2.ps1 && .\\build.ps1 && .\\run_dual_sync.ps1" >&2
  echo "WSL2 remains an alternate. local_approx_sync.py is NOT ROS." >&2
  exit 1
}

if ! command -v colcon >/dev/null 2>&1; then
  fail "colcon not found on PATH. Install ROS2 / use WSL2, then re-run this script."
fi

# Source a ROS distro if the shell was not already set up.
if [[ -z "${AMENT_PREFIX_PATH:-}" ]]; then
  for distro in jazzy humble iron; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      # shellcheck disable=SC1090
      source "/opt/ros/${distro}/setup.bash"
      echo "Sourced /opt/ros/${distro}/setup.bash"
      break
    fi
  done
fi

if [[ -z "${AMENT_PREFIX_PATH:-}" ]]; then
  fail "ROS2 overlay not sourced (AMENT_PREFIX_PATH empty) and no /opt/ros/*/setup.bash found."
fi

if [[ ! -f "$ROOT/src/orbbec_dual_sync/package.xml" ]]; then
  fail "Missing package at src/orbbec_dual_sync (run from Camera/ros2_ws)."
fi

if [[ ! -f "$ROOT/src/orbbec_dual_sync/resource/orbbec_dual_sync" ]]; then
  fail "Missing ament resource index marker: src/orbbec_dual_sync/resource/orbbec_dual_sync"
fi

echo "Building orbbec_dual_sync (ROS_DISTRO=${ROS_DISTRO:-unknown})..."
colcon build --packages-select orbbec_dual_sync --symlink-install

echo ""
echo "Build OK — real ROS2 Nodes are installed under install/."
echo "Next (required every new shell):"
echo "  source $ROOT/install/setup.bash"
echo "  export ORBBEC_CAMERA_ROOT=\$(cd $ROOT/.. && pwd)   # Camera/ with depthcamera.py"
echo "  ros2 launch orbbec_dual_sync dual_sync_detect.launch.py"
echo ""
echo "Config (YAML loaded by launch):"
echo "  $ROOT/install/orbbec_dual_sync/share/orbbec_dual_sync/config/dual_sync.yaml"
echo "Nodes: orbbec_cam0_pub, orbbec_cam1_pub, orbbec_sync_detect (rclpy Node subclasses)"
echo "Do NOT use python local_approx_sync.py as the sync workflow — that is NOT ROS."
