# Camera project retrospective

## What worked

- **Define a race, then measure it.** The Best-of-N sideways-vial race gave every
  method the same positive/negative set and ranked recall, false positives, and
  IoU explicitly. `B2_improved` was a useful stepping stone; `I_hard_robust`
  ultimately won the expanded set at 38/40 recall, 0/36 false positives, and
  0.8448 mean IoU.
- **Pin the evaluated winner at the integration boundary.** ROS imports
  `I_hard_robust` directly instead of silently following whichever experiment
  `depthcamera.py` happens to expose. Keep that pin, but keep it inside the
  isolated detector worker so an experimental detector failure cannot kill
  capture or ROS synchronization.
- **Separate capture from detection.** The two ROS publishers are stream-only:
  no YOLO/Torch load and no inference. The sync node hands pairs to a subprocess
  detector. A native detector abort can be reported and restarted without
  taking down camera capture.
- **Make Windows ROS startup boring.** `_ros_env.cmd` plus `build.cmd` and
  `launch_dual_sync.cmd` reliably activate Pixi/RoboStack and the install
  overlay. This is better than asking users to translate Linux `source`
  instructions into PowerShell. In PowerShell, run `.\launch_dual_sync.cmd` or
  `.\run_dual_sync.ps1` (one-shot — no separate `source_install`). Bare
  `source_install.cmd` fails; use `.\source_install.cmd` or `.\source_install.ps1`.
- **Publisher `preview=off` vs sync preview ON.** Raw USB preview stays off on
  `orbbec_cam*_pub` by design; the default UI is `orbbec_sync_detect` (`preview:=true`).
- **Serialize and stagger dual USB opens.** A cross-process Windows lock,
  delayed cam1 start, retries, and publisher respawn reduced CAP_OBSENSOR races.
  Each Orbbec still needs a real USB 3 port; software cannot fix a shared,
  bandwidth-starved hub.
- **Use tkinter for preview.** Pillow/tkinter displays BGR frames after
  conversion without touching OpenCV HighGUI. Closing the preview leaves ROS
  alive. OpenCV `imshow` remains an explicit opt-in only.
- **Prefer latest-frame queues.** Best-effort DDS with shallow queues and a
  one-pair detector worker avoids accumulating stale RGB-D frames and native
  memory pressure.

## What did not work

- **YOLO inside a Windows ROS launch child.** The Torch/Ultralytics and
  OpenCV/RoboStack runtime combination can abort natively on first inference
  with exit code 3. Python `try/except` cannot catch a native process abort.
  YOLO must remain off by default until the runtime conflict is proven fixed;
  subprocess isolation and respawn are required even when it is enabled.
- **Disabling preview as “hardening.”** It stopped UI crashes, but also made a
  healthy launch look broken: the user saw no cameras. Stability is not useful
  if the main success criterion is invisible. The synchronized tkinter preview
  is now on by default; only headless launches should pass `preview:=false`.
- **Probing HighGUI in a launch child.** `namedWindow`/`imshow` may terminate the
  process before an exception is raised. Do not use a “try it and fall back”
  strategy there. Default directly to tkinter.
- **Mixing winner promotion with crash repair.** Pinning a detector winner and
  changing process/runtime behavior at the same time made regressions hard to
  attribute. Promote the algorithm in the worker, then independently verify
  worker crash containment, publisher uptime, and preview.
- **Methods C/E intrinsics assumptions.** Experimental methods that treated
  intrinsics as a fixed tuple/dictionary shape hit `KeyError` and integration
  mismatches. Normalize the dataset contract once at the evaluator boundary
  and test missing intrinsics explicitly.
- **H overfit the old synthetic set.** Its old-set score looked excellent, but
  recall fell to 0.725 on the hard-expanded set. Never promote from one synthetic
  generation; retain old and hard sets and report both.
- **Opening both USB cameras together.** Concurrent CAP_OBSENSOR initialization
  was unreliable. More retries alone did not remove the race; serialization and
  staggered startup were needed.

## What future prompts should say

Use concrete, ordered acceptance criteria. A good prompt is:

> ROS2-only change. Do not edit `depthcamera.py` or `sideways_race`. On
> `launch_dual_sync.cmd`, both live cameras must appear in one annotated preview.
> Keep publishers and the sync node alive if preview closes or detection dies.
> Keep YOLO off by default, detector subprocess isolation, respawn, and the
> pinned `I_hard_robust` winner. Hardware is two Orbbecs on separate USB 3 ports.
> Rebuild the install tree, smoke-test tkinter with fake frames if hardware is
> unavailable, and report the exact relaunch command. Evaluate detection with
> recall@0.5, FP count/rate, mean IoU, sync delta, and process exit behavior.

Also state:

- whether the target is ROS2 launch, standalone Python, or both;
- which files are in/out of scope;
- whether a visible preview is mandatory and whether headless mode matters;
- hardware/OS/runtime constraints and whether cameras are available to test;
- process-liveness requirements after UI close, camera loss, and detector crash;
- the baseline and exact metrics that must not regress.

## Prompt anti-patterns

- “Fix it” without naming the failing command, expected screen output, logs, or
  success criteria.
- Asking to disable all UI for stability and later assuming a camera window will
  appear.
- Combining algorithm research, ROS packaging, USB recovery, UI redesign, and
  runtime upgrades without a priority order.
- Requesting broad rewrites while also requiring existing crash hardening and
  detector behavior to remain identical.
- Saying “improve accuracy” without a frozen dataset, negative set, IoU
  threshold, false-positive budget, and baseline.
- Treating “no exception” as success when a native child can exit or a preview
  can silently disable itself.
- Assuming a software change can overcome two cameras on a shared USB2 path.

The practical priority order is: **live camera streams visible, processes stay
alive, synchronized frames arrive, detection works, then accuracy experiments.**
