# Research Supervisor — Result

**Date:** 2026-10-02  
**Repo:** `E:\Desktop\Jessica\Camera`

## On-task checklist

| Agent | Theme | File | plan.md section | Citations/URLs | On-task |
|------:|-------|------|-----------------|:--------------:|:-------:|
| 01 | Late fusion / detection voting | `agent_01.md` | WBF late fusion | yes | yes |
| 02 | Epipolar / cross-view consistency | `agent_02.md` | Isaac-Medina epipolar | yes | yes |
| 03 | RGB-D transparent / depth-aware | `agent_03.md` | MVTrans (+ TransCG/DFNet) | yes | yes |
| 04 | Extrinsics + triangulation | `agent_04.md` | ChArUco + Hartley–Sturm | yes | yes |
| 05 | MCMOT / tracklet fusion | `agent_05.md` | ReST | yes | yes |
| 06 | Cross-camera ReID | `agent_06.md` | TransReID | yes | yes |
| 07 | Domain gen / multi-view aug | `agent_07.md` | MVAug | yes | yes |
| 08 | Transparent object literature | `agent_08.md` | MVTrans (+ ClearGrasp/TransLab) | yes | yes |
| 09 | Kalman / Bayesian dual-cam | `agent_09.md` | UKF multi-view fusion | yes | yes |
| 10 | Transformer multi-view detectors | `agent_10.md` | MVDeTr | yes | yes |

**Operational content preserved:** BoN #3 asymmetric gates, `locate_pair`, Attempt E / `bon-winner`, clear vial + white cap, no Mask R-CNN/SAM mandate, dataset recovery notes. `vial_dataset` not wiped.

## Repairs made

- All 10 agent writeups were already present in `plan.md` (appended out of order by parallel agents).
- **Reordered** Research Agent sections to **01→10**.
- **Appended** `## Research Supervisor — Synthesis`.
- No content restore from git needed (BoN merge did not wipe research or operational sections).
- No agent file content was replaced; plan summaries kept as written by agents.

## Top-3 ranking (this dual Orbbec clear-vial setup)

1. **Agent 02 — Epipolar peer gating (Isaac-Medina)** — gate `locate_pair` soft-retry with \(F\)/cap-center consistency; no retrain; keep per-cam distances.
2. **Agent 01 — WBF / dual-score vote** — same-tick score consensus after white-cap 3D associate; fuse ID/score only; keep BoN #3 gates.
3. **Agent 04 — ChArUco extrinsics + Hartley–Sturm triangulation** — enables (1) and optional world XYZ of lid; overlay cm stays per-camera.

**Honorable next (train-time):** Agent 07 MVAug mild perspective augs after dual recapture.

**Deferred / conflict notes:** MVTrans (03/08), MVDeTr (10), ReST (05), TransReID (06) need more data/calib; UKF fused distance (09) conflicts with “never average distances” unless optional labeled world overlay.

## Confirmation

- [x] `plan.md` contains all 10 `## Research Agent NN` sections (ordered 01–10)
- [x] `plan.md` ends with `## Research Supervisor — Synthesis`
- [x] All `plan_research/agent_01.md` … `agent_10.md` present with citations
- [x] BoN / `locate_pair` / asymmetric gates text preserved
