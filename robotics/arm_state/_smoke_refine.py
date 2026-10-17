from pathlib import Path
import cv2
from orbbec_dual_sync.arm_overlay_draw import get_worker, default_arm_weights
w = default_arm_weights(Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics"))
img = cv2.imread(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\live_for_slots.png")
wrk = get_worker(w, 0.25)
raw = wrk.predict(img, refine=False)
ref = wrk.predict(img, cam="cam0", refine=True)
print("raw", raw)
print("ref", ref)
ov = img.copy()
for xyxy, c in ref:
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    cv2.rectangle(ov, (x1,y1), (x2,y2), (255,220,0), 2)
    cv2.putText(ov, f"arm {c:.2f}", (x1, max(16,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,0), 2)
cv2.imwrite(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_state\preview_cam0_refined.png", ov)
print("wrote preview_cam0_refined.png")
