import cv2
import numpy as np
from pathlib import Path

src = Path(r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-0730b70c-1d0b-4831-bdbb-2296e6918cfb.png")
out_dir = Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state")
img = cv2.imread(str(src))
# Orange border: high R, low-mid B
b,g,r = cv2.split(img)
orange = (r > 180) & (g > 60) & (g < 200) & (b < 120)
# Content = not orange
content = (~orange).astype(np.uint8) * 255
ys, xs = np.where(content > 0)
print("content bbox", xs.min(), ys.min(), xs.max(), ys.max())
pad = 1
x0,y0 = int(xs.min())+pad, int(ys.min())+pad
x1,y1 = int(xs.max())-pad, int(ys.max())-pad
crop = img[y0:y1+1, x0:x1+1]
# Match open template aspect roughly 63x50
open_t = cv2.imread(str(out_dir / "template_furnace.png"))
th, tw = open_t.shape[:2]
resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
(out_dir / "templates").mkdir(exist_ok=True)
cv2.imwrite(str(out_dir / "template_closed.png"), resized)
cv2.imwrite(str(out_dir / "templates" / "template_closed.png"), resized)
cv2.imwrite(str(out_dir / "seed_closed.png"), resized)
# Also keep raw content crop
cv2.imwrite(str(out_dir / "seed_closed_raw.png"), crop)
print("saved closed", resized.shape, "from content", crop.shape)
