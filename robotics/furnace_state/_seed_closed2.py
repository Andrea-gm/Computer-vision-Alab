import cv2
import numpy as np
from pathlib import Path
src = Path(r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-0730b70c-1d0b-4831-bdbb-2296e6918cfb.png")
out = Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state")
img = cv2.imread(str(src))
# Fixed inset to drop orange frame (works on small overlay crops)
inset = 5
crop = img[inset:-inset, inset:-inset]
open_t = cv2.imread(str(out/"template_furnace.png"))
th,tw = open_t.shape[:2]
closed = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(out/"template_closed.png"), closed)
cv2.imwrite(str(out/"templates"/"template_closed.png"), closed)
cv2.imwrite(str(out/"seed_closed.png"), closed)
print("closed saved", closed.shape)

# Score closed tmpl vs open tmpl (should be lower if different)
def ncc(a,b):
    ag=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ag=(ag-ag.mean())/(ag.std()+1e-6)
    bg=(bg-bg.mean())/(bg.std()+1e-6)
    return float((ag*bg).mean())
print("ncc open vs closed template", ncc(open_t, closed))

# Score against slot crops on latest capture / preview
caps = sorted((out/"captures").glob("cam0_*.png"), key=lambda p:p.stat().st_mtime, reverse=True)
print("n captures", len(caps))
if caps:
    frame=cv2.imread(str(caps[0]))
    import json
    cfg=json.loads((out/"furnace.json").read_text(encoding="utf-8-sig"))
    for s in cfg["slots"]:
        x,y,w,h=s["x"],s["y"],s["w"],s["h"]
        roi=frame[y:y+h,x:x+w]
        if roi.size==0: continue
        roi_r=cv2.resize(roi,(tw,th))
        print(f"slot{s['id']} ncc_open={ncc(roi_r,open_t):.3f} ncc_closed={ncc(roi_r,closed):.3f}")
