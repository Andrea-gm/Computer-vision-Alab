import sys
sys.path.insert(0, r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src\orbbec_dual_sync")
from orbbec_dual_sync.arm_overlay_draw import get_worker, default_arm_weights
from pathlib import Path
import cv2
w = Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_state\runs\arm_detect\weights\best.pt")
img = cv2.imread(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\live_for_slots.png")
boxes = get_worker(w, conf=0.25).predict(img, conf=0.25)
print("boxes", len(boxes), boxes[:3])
