import cv2
from pathlib import Path
closed = Path(r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-0730b70c-1d0b-4831-bdbb-2296e6918cfb.png")
img = cv2.imread(str(closed))
print("closed", img.shape if img is not None else None)
for name in ["template_furnace.png", "seed_crop.png", "templates/furnace0.png", "templates/furnace1.png"]:
    p = Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state") / name
    im = cv2.imread(str(p))
    print(name, im.shape if im is not None else None)
