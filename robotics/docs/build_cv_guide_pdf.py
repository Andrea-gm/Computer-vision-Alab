"""Build a ~10-page PDF guide to the Alab dual-Orbbec computer vision stack."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent / "Alab_Computer_Vision_Guide.pdf"


def _pdf_text(s: str) -> str:
    """Core Helvetica is Latin-1 only; normalize fancy punctuation."""
    return (
        s.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("\u2192", "->")
        .replace("\u00d7", "x")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


class GuidePDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="Letter", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "Alab Dual-Orbbec Computer Vision Guide", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(18, 14, 198, 14)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(
            0,
            8,
            "JessN777/Computer-vision-Alab  |  scene+tape extrinsics PR  |  for operators & developers",
            align="C",
        )

    def h1(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 40, 70)
        self.multi_cell(0, 9, _pdf_text(text))
        self.set_x(self.l_margin)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def h2(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 100)
        self.multi_cell(0, 7, _pdf_text(text))
        self.set_x(self.l_margin)
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def h3(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, _pdf_text(text))
        self.set_x(self.l_margin)
        self.set_text_color(0, 0, 0)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.2, _pdf_text(text))
        self.set_x(self.l_margin)
        self.ln(1.5)

    def code(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 248)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, _pdf_text(text), fill=True)
        self.set_x(self.l_margin)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 10)
        self.ln(1.5)

    def callout(self, title: str, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_fill_color(235, 243, 255)
        self.set_draw_color(60, 100, 160)
        self.set_font("Helvetica", "B", 9)
        self.multi_cell(0, 5, _pdf_text(title), fill=True)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.8, _pdf_text(text), fill=True)
        self.set_x(self.l_margin)
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.0, _pdf_text(f"  -  {text}"))
        self.set_x(self.l_margin)


def build() -> Path:
    pdf = GuidePDF()

    # ----- Page 1: Title + map -----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 35, 65)
    pdf.ln(20)
    pdf.multi_cell(
        0,
        10,
        _pdf_text("Understanding the Alab\nDual-Camera Computer Vision System"),
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0,
        6,
        _pdf_text(
            "A practical guide to what the cameras see, how space is represented,\n"
            "how the purple tape and blue joints are used, and what to run today."
        ),
    )
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5.5,
        _pdf_text(
            "Repository: github.com/JessN777/Computer-vision-Alab\n"
            "Workspace layout: robotics/ (Python CV tools) + ros2_ws/ (ROS 2 dual_view stack)\n"
            "Hardware: two Orbbec Gemini-class RGB-D cameras + UR arm cell\n"
            "Audience: you - operator / developer trying to see the big picture"
        ),
    )
    pdf.ln(8)
    pdf.h2("What this guide covers")
    for line in [
        "1. Big picture: two eyes on one cell",
        "2. Hardware and software map",
        "3. Coordinate frames (ur_base, cameras, purple tape)",
        "4. Why tape is not ArUco — and what we do instead",
        "5. Scene + purple-tape extrinsics (new path)",
        "6. Live dual_view: furnace, rack, blue joints, Kalman",
        "7. Vial / depthcamera stack (separate but related)",
        "8. Files that matter and what \"ready\" means",
        "9. How to run things day-to-day",
        "10. Troubleshooting and what \"good\" looks like",
    ]:
        pdf.bullet(line)

    # ----- Page 2 -----
    pdf.add_page()
    pdf.h1("1. Big picture: two eyes on one cell")
    pdf.body(
        "The lab cell has a Universal Robots (UR) arm, a rack, a furnace area, and other "
        "equipment. Two Orbbec RGB-D cameras are mounted at roughly right angles so they "
        "look into the same workspace from different sides. Each camera gives a color "
        "image and a depth map (millimeters to each pixel on opaque surfaces)."
    )
    pdf.body(
        "Computer vision here is not one monolith. It is several cooperating jobs:"
    )
    pdf.bullet("Watch the arm: find blue joint blobs, keep IDs when they leave the frame.")
    pdf.bullet("Know where \"home\" is: purple tape / pedestal as a shared origin.")
    pdf.bullet("Know where each camera sits in the cell (extrinsics / cell_frames).")
    pdf.bullet("Overlay furnace and rack state (OPEN/CLOSED style ROIs).")
    pdf.bullet("Optionally track clear vials with white caps (depthcamera / YOLO path).")
    pdf.callout(
        "Mental model",
        "Think of cam0 and cam1 as two people standing at the corners of a room. "
        "They see the same furniture from different angles. If you know how those "
        "two people are oriented relative to each other (and to the robot base), "
        "you can say where an object is in one shared room coordinate system — "
        "called ur_base in this project.",
    )
    pdf.h2("What \"spatially understanding\" means")
    pdf.body(
        "For a point seen by a camera at pixel (u,v) with depth Z millimeters, the "
        "camera can convert that to a 3D point in its own frame (X right, Y down, "
        "Z forward — OpenCV convention). To compare cam0 and cam1, or to talk about "
        "the robot base, you need a rigid transform: rotation R and translation t."
    )
    pdf.code("p_base = R @ p_cam + t_mm     # cell_frames.py")
    pdf.body(
        "If R and t are wrong, the two cameras disagree about where the same physical "
        "joint is. That is why purple tape, scene matching, and (optionally) a "
        "calibration board all exist: they are ways to get R and t right."
    )

    # ----- Page 3 -----
    pdf.add_page()
    pdf.h1("2. Hardware and software map")
    pdf.h2("Hardware")
    pdf.bullet("Two Orbbec RGB-D cameras (USB 3 — each on its own port when possible).")
    pdf.bullet("UR arm with blue-highlighted joints (HSV \"baby blue\" blobs).")
    pdf.bullet("Purple tape near the pedestal / base — shared visual origin.")
    pdf.bullet("Rack and furnace regions used by overlay classifiers.")
    pdf.h2("Software trees")
    pdf.body(
        "Computer-vision-Alab splits into two main folders after the Jessica split plan:"
    )
    pdf.code(
        "Computer-vision-Alab/\n"
        "  robotics/          # plain Python: calib, vial tools, cell_frames.json\n"
        "  ros2_ws/           # ROS 2 + Pixi: orbbec_dual_sync dual_view live stack"
    )
    pdf.h2("Key live ROS package: orbbec_dual_sync")
    pdf.bullet("camera_publisher — publish RGB/depth from each Orbbec.")
    pdf.bullet("dual_view_node — side-by-side preview: furnace | rack + arm overlays.")
    pdf.bullet("baby_blue_draw — blue joint / purple tape HSV helpers.")
    pdf.bullet("blue_joint_tracks — 6-slot joint bank + Kalman coast in ur_base.")
    pdf.bullet("cell_frames — load/save camera poses in the cell.")
    pdf.bullet("joint_motion_net — small motion net to keep predicted boxes on screen.")
    pdf.h2("Key robotics (non-ROS) tools")
    pdf.bullet("scene_extrinsics_calib.py — NEW: ArUco-free R,t from scene + tape.")
    pdf.bullet("stereo_calib_stub.py — classic ChArUco/chessboard path (optional).")
    pdf.bullet("depthcamera.py — vial distance / dual-independent locate.")
    pdf.bullet("cell_frames.json / stereo_extrinsics.json — saved geometry.")
    pdf.callout(
        "Environment tip",
        "ROS launches expect ORBBEC_CAMERA_ROOT to point at the robotics/ folder "
        "so nodes can import depthcamera / read JSON configs. If imports fail, "
        "check that variable first.",
    )

    # ----- Page 4 -----
    pdf.add_page()
    pdf.h1("3. Coordinate frames (what is \"where\")")
    pdf.h2("ur_base — the shared world")
    pdf.body(
        "The world origin (0,0,0) is defined at the robot base / blue pedestal idea. "
        "In the live stack, purple tape is treated as the shared XY origin for joint "
        "polar coordinates (distance and angle from the tape). cell_frames.json stores "
        "each camera's R and t_mm so that camera points can be mapped into ur_base."
    )
    pdf.h2("Camera optical frames")
    pdf.body(
        "Each Orbbec has its own frame: +X to the right in the image, +Y down, +Z "
        "out of the lens. Depth values are along +Z for that camera. Clear plastic "
        "(vial cups) often returns bad depth; opaque lids and blue joints are more "
        "trustworthy."
    )
    pdf.h2("Current default layout: perpendicular_90")
    pdf.body(
        "Today's cell_frames.json uses layout=perpendicular_90: cam0 has identity "
        "rotation; cam1 is assumed yawed +90 degrees about Y (cameras looking inward "
        "at right angles). Translations t_mm were bootstrapped so the pedestal maps "
        "near the origin. That is a strong prior — often good enough to start — but "
        "it is not a full measured calibration."
    )
    pdf.code(
        "layout: perpendicular_90\n"
        "cam0: R = I,  t bootstrapped from pedestal\n"
        "cam1: R = yaw+90, t bootstrapped from pedestal"
    )
    pdf.h2("After scene+tape calib: layout=scene_tape")
    pdf.body(
        "When you run scene_extrinsics_calib.py successfully, the layout name becomes "
        "scene_tape. The assumed 90 degree R is replaced by a solve from real matched "
        "3D points. Purple tape anchors the origin in both views."
    )
    pdf.h3("Stereo file vs cell_frames")
    pdf.bullet("stereo_extrinsics.json — maps points from cam0 into cam1: X1 = R X0 + t.")
    pdf.bullet("cell_frames.json — maps each camera into ur_base (robot-centric).")
    pdf.body(
        "They describe the same geometry in different packaging. Scene calib writes "
        "both so vial reproject assist and dual_view joint fusion can stay consistent."
    )

    # ----- Page 5 -----
    pdf.add_page()
    pdf.h1("4. Purple tape, ArUco, and what failed")
    pdf.h2("What you tried with tape")
    pdf.body(
        "Colored tape is easy to put in the cell. The program can find a purple blob "
        "with HSV thresholds. That works well for \"where is the origin spot?\" It "
        "works poorly if you ask the program to read the full 3D orientation of a "
        "short shiny strip the way ArUco corners would."
    )
    pdf.h2("What ArUco / ChArUco would give you")
    pdf.body(
        "A printed board with unique corner IDs gives many known 3D–2D correspondences "
        "in both cameras. OpenCV stereoCalibrate then returns accurate R and t. That "
        "path still exists (stereo_calib_stub.py) but is hard when the arm fills the "
        "overlap and you cannot place a board."
    )
    pdf.h2("What purple tape IS good for")
    pdf.bullet("Centroid: a named point both cameras can agree is \"home\" (0,0).")
    pdf.bullet("Long axis: the strip is a LINE — that helps lock yaw (heading), not full 6-DoF alone.")
    pdf.bullet("Joint polars: r_mm and ang_deg of blue joints relative to the tape origin.")
    pdf.h2("What purple tape is NOT")
    pdf.bullet("Not a substitute for many scene point matches when solving full camera R,t.")
    pdf.bullet("Not reliable if you treat strip \"which way is up\" as a precise pose by itself.")
    pdf.bullet("Not required for every feature — pedestal blue base can also bootstrap t.")
    pdf.callout(
        "Rule of thumb",
        "Use tape as a bright landmark (point + line). Use the rest of the static "
        "scene (rack edges, fixtures) as many point pairs. Together they replace ArUco "
        "without needing a printed board in the arm's way.",
    )

    # ----- Page 6 -----
    pdf.add_page()
    pdf.h1("5. Scene + purple-tape extrinsics (current recommended path)")
    pdf.body(
        "This is the ArUco-free path added in PR #1 (feat/scene-tape-extrinsics). "
        "It answers: \"Given what both cameras already see from different angles, "
        "where is cam1 relative to cam0?\""
    )
    pdf.h2("Algorithm (plain language)")
    pdf.bullet("1. Grab synced RGB + depth from both cameras.")
    pdf.bullet("2. Mask out blue arm pixels so moving links are not used as landmarks.")
    pdf.bullet("3. Detect ORB features on static structure; match descriptors across views.")
    pdf.bullet("4. For each match, read depth and unproject to 3D millimeters.")
    pdf.bullet("5. Find purple tape in both views: centroid + long-axis direction.")
    pdf.bullet("6. Weight tape center (and axis endpoints) heavily in the point cloud.")
    pdf.bullet("7. RANSAC + Umeyama: find rigid R,t mapping cam0 points into cam1.")
    pdf.bullet("8. Refine yaw so transformed tape axis aligns with the other camera's axis.")
    pdf.bullet("9. Write stereo_extrinsics.json and update cell_frames (layout=scene_tape).")
    pdf.h2("Commands")
    pdf.code(
        "cd robotics\n"
        "python scene_extrinsics_calib.py howto\n"
        "python scene_extrinsics_calib.py solve --live\n"
        "  # SPACE = buffer a dual frame, S = solve + write, Q = quit\n"
        "python scene_extrinsics_calib.py status\n"
        "python -m pytest test_scene_extrinsics.py -v"
    )
    pdf.h2("When the solve fails")
    pdf.bullet("Not enough overlap / no texture — point cameras more into shared structure.")
    pdf.bullet("Arm filling the view — park it; masking helps but empty static FOV is better.")
    pdf.bullet("Tape missing in one camera — fix lighting / HSV; longer matte purple strip helps.")
    pdf.bullet("Depth holes — prefer opaque surfaces; avoid trusting glass.")
    pdf.body(
        "Success looks like: status shows non-PLACEHOLDER stereo notes, layout=scene_tape, "
        "and the same physical landmark (tape center) maps near the same ur_base XYZ "
        "from both cameras."
    )

    # ----- Page 7 -----
    pdf.add_page()
    pdf.h1("6. Live dual_view: what you see on screen")
    pdf.body(
        "dual_view_node builds a side-by-side preview. It subscribes to camera topics "
        "(it does not open CAP_OBSENSOR itself). Left pane is typically furnace-oriented "
        "CV; right pane rack OPEN/CLOSED style overlays. Arm / joint drawing is layered "
        "using baby_blue helpers and the joint bank."
    )
    pdf.h2("Blue joints + Kalman coast")
    pdf.body(
        "Blue joint detections are HSV blobs with depth lift into camera frame, then "
        "into ur_base via cell_frames. A fixed-size bank (default 6 tracks) associates "
        "detections over time. When a joint leaves the camera, Kalman prediction keeps "
        "the highlight and angle label for several seconds (\"coast\"). A small motion "
        "network can also help predicted boxes stay plausible."
    )
    pdf.h2("Purple tape in the live stack")
    pdf.body(
        "find_purple_tape returns a box (+ optional depth). Joint tracks report polar "
        "coordinates relative to that origin: distance in the XY plane and yaw angle. "
        "If the tape flickers, a short hold keeps the last origin. Scene calib extends "
        "this idea with an explicit long-axis measurement for orientation locking."
    )
    pdf.h2("Furnace and rack overlays")
    pdf.body(
        "These are mostly ROI / color / geometry overlays configured by JSON (furnace "
        "and rack configs under robotics or package share). They answer operational "
        "questions (e.g. rack closed) and are separate from the extrinsics solve — "
        "but they share the same camera images."
    )
    pdf.callout(
        "Independence note",
        "Joint tracking can look \"alive\" even when stereo_extrinsics.json is still a "
        "PLACEHOLDER, because perpendicular_90 + pedestal bootstrap already provide a "
        "usable ur_base. Scene calib improves agreement between cameras; it is not "
        "required for every overlay to draw.",
    )

    # ----- Page 8 -----
    pdf.add_page()
    pdf.h1("7. Vial / depthcamera stack (related sibling)")
    pdf.body(
        "Separately from dual_view arm overlays, robotics/depthcamera.py tracks clear "
        "vials with white screw caps. Clear cups break depth; distance is taken from "
        "the opaque white lid. A YOLO model proposes boxes; classical gates "
        "(clear-vial hole under the lid, scale at Z, sideways assist) accept or reject."
    )
    pdf.h2("Dual vial modes")
    pdf.bullet("--dual-independent (preferred): two processes; localhost IPC only corroborates cm.")
    pdf.bullet("--dual / locate_pair: older sticky peer soft-assist path.")
    pdf.bullet("--3d-reproject-assist: if stereo_extrinsics is REAL, propose a box on the misser.")
    pdf.body(
        "Scene+tape calib is exactly what makes --3d-reproject-assist trustworthy. While "
        "stereo_extrinsics.json says PLACEHOLDER, reproject assist refuses to invent boxes."
    )
    pdf.h2("How this connects to the arm cell")
    pdf.body(
        "Same two cameras, different jobs. Arm dual_view cares about joints and cell "
        "frames. Vial tools care about white caps and per-camera centimeters. Shared "
        "geometry files (stereo_extrinsics / cell_frames) are the bridge when you want "
        "cross-camera geometric assist."
    )

    # ----- Page 9 -----
    pdf.add_page()
    pdf.h1("8. Files that matter right now")
    pdf.h3("Geometry state")
    pdf.bullet("robotics/cell_frames.json — ur_base poses; layout perpendicular_90 or scene_tape.")
    pdf.bullet("robotics/stereo_extrinsics.json — cam0→cam1; PLACEHOLDER until calibrated.")
    pdf.h3("Calib tools")
    pdf.bullet("robotics/scene_extrinsics.py — library: ORB, tape axis, Umeyama/RANSAC.")
    pdf.bullet("robotics/scene_extrinsics_calib.py — CLI live/file solve.")
    pdf.bullet("robotics/stereo_calib_stub.py — optional board path + howto pointer.")
    pdf.bullet("robotics/test_scene_extrinsics.py — unit tests (no cameras needed).")
    pdf.h3("Live dual_view")
    pdf.bullet("ros2_ws/.../dual_view_node.py — main preview node.")
    pdf.bullet("ros2_ws/.../blue_joint_tracks.py — joint bank + tape origin polars.")
    pdf.bullet("ros2_ws/.../baby_blue_draw.py — HSV blue + purple tape (+ long-axis helper).")
    pdf.bullet("ros2_ws/.../cell_frames.py — ROS import of the frame graph.")
    pdf.bullet("robotics/joint_motion_mlp.npz — weights for motion coast assist.")
    pdf.h3("Vial (optional)")
    pdf.bullet("robotics/depthcamera.py — live vial distance.")
    pdf.bullet("robotics/vial_3d_model.py — dims + extrinsics loaders.")
    pdf.h2("GitHub status (as of this guide)")
    pdf.body(
        "Canonical repo: JessN777/Computer-vision-Alab. Scene+tape work was pushed via "
        "fork Andrea-gm and opened as Pull Request #1 because collaborator accounts "
        "had pull-only access to main. Merge that PR (or grant write access) to land "
        "scene_extrinsics_* on main."
    )

    # ----- Page 10 -----
    pdf.add_page()
    pdf.h1("9. Day-to-day runbook")
    pdf.h2("A. Improve camera spatial agreement (recommended)")
    pdf.code(
        "1. Park or slow the UR arm.\n"
        "2. Ensure purple tape is visible to BOTH Orbbecs.\n"
        "3. cd robotics\n"
        "4. python scene_extrinsics_calib.py solve --live\n"
        "5. SPACE a few varied dual frames, then S.\n"
        "6. python scene_extrinsics_calib.py status"
    )
    pdf.h2("B. Live dual_view overlays")
    pdf.body(
        "Use your usual launch scripts under ros2_ws (Pixi + launch_*.cmd patterns). "
        "Set ORBBEC_CAMERA_ROOT to the robotics folder. Start camera publishers, then "
        "dual_view. You should see furnace/rack panes and blue joint highlights with "
        "angles relative to the purple origin when tape is found."
    )
    pdf.h2("C. Vial distance (optional)")
    pdf.code(
        "cd robotics\n"
        "python depthcamera.py --dual-independent\n"
        "python depthcamera.py --dual-independent --3d-reproject-assist"
    )
    pdf.h1("10. Troubleshooting cheat sheet")
    pdf.h3("Joints jump between cameras / disagree in 3D")
    pdf.body("Re-run scene+tape calib; verify cell_frames ready flags; check tape visibility.")
    pdf.h3("Purple tape not found")
    pdf.body(
        "Lighting and HSV band (magenta). Try a longer matte purple strip. Avoid strong "
        "red clutter near the pedestal. Use _hsv_probe helpers under arm_state if needed."
    )
    pdf.h3("Reproject assist does nothing")
    pdf.body(
        "stereo_extrinsics still PLACEHOLDER — run scene_extrinsics_calib or board calib "
        "until status is clean."
    )
    pdf.h3("Imports / ROS cannot find robotics")
    pdf.body("Export ORBBEC_CAMERA_ROOT to the absolute path of robotics/.")
    pdf.h3("What \"good\" looks like")
    pdf.bullet("Tape visible; joint labels stable; coast works when a joint leaves view.")
    pdf.bullet("status: non-placeholder stereo; cell_frames layout=scene_tape after calib.")
    pdf.bullet("Same landmark within a few centimeters from both cams in ur_base.")
    pdf.ln(4)
    pdf.callout(
        "One-sentence summary",
        "Two depth cameras watch the UR cell; purple tape marks home; blue blobs are "
        "joints; cell_frames puts everyone in ur_base; scene+tape matching (not ArUco) "
        "is how you teach the computers where they sit relative to each other when the "
        "arm is in the way.",
    )
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0,
        5,
        _pdf_text(
            "End of guide. Rebuild this PDF anytime with:\n"
            "  python robotics/docs/build_cv_guide_pdf.py"
        ),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
    try:
        from pypdf import PdfReader

        print(f"Pages: {len(PdfReader(str(path)).pages)}")
    except Exception:
        # fpdf page count
        print("Open the PDF to confirm page count (target ~10).")
