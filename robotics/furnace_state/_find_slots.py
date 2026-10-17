import cv2
import numpy as np
from pathlib import Path

# Reference image with all furnaces
ref = Path(r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-19156ec1-3552-4996-92e4-96d47f879b00.png")
img = cv2.imread(str(ref))
print("ref shape", None if img is None else img.shape)
# Also false detection crop
bad = Path(r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-8c4d2e59-1220-43b1-beff-c2c1aa58d531.png")
bimg = cv2.imread(str(bad))
print("bad shape", None if bimg is None else bimg.shape)

# Current open template match across full ref to propose 4 peaks
tmpl = cv2.imread(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\template_furnace.png")
closed = cv2.imread(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\template_closed.png")
print("tmpl", tmpl.shape if tmpl is not None else None, "closed", closed.shape if closed is not None else None)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Try both templates
for name, t in [("open", tmpl), ("closed", closed)]:
    if t is None: continue
    tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    th, tw = tg.shape
    res = cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED)
    # Find peaks with NMS
    peaks = []
    res2 = res.copy()
    for i in range(8):
        minv, maxv, minl, maxl = cv2.minMaxLoc(res2)
        if maxv < 0.35: break
        x,y = maxl
        peaks.append((x,y,tw,th,float(maxv)))
        # zero neighborhood
        y0=max(0,y-th//2); y1=min(res2.shape[0], y+th//2)
        x0=max(0,x-tw//2); x1=min(res2.shape[1], x+tw//2)
        res2[y0:y1, x0:x1] = 0
    peaks = sorted(peaks, key=lambda p: p[0])
    print(name, "peaks", peaks)
