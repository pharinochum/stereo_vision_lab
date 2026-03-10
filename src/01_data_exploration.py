# ────────────────────────────────────────────────────────────────
# 01_data_exploration.ipynb
# Goal: Understand the dataset structure, file naming, and visualize
#       raw images + ground truth maps
# ────────────────────────────────────────────────────────────────

# === Cell 1: Imports & constants ===

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    load_stereo_pair,
    load_gt_map,
    disparity_to_depth,
    BASELINE_CM, FOCAL_PX
)

sns.set_style("whitegrid")

ILLUM = "daylight"      # can be: daylight, fluorescent, lamps, flashlight
FRAME  = 1            # try different frames: 0, 300, 600, 1200, ...

# === Cell 2: Load one stereo pair ===
left, right = load_stereo_pair(illum=ILLUM, frame=FRAME)

print(f"Left  image shape: {left.shape}  dtype: {left.dtype}")
print(f"Right image shape: {right.shape} dtype: {right.dtype}")

# === Cell 3: Load ground truth maps ===
disp_gt   = load_gt_map(FRAME, side="L", map_type="disparity")
depth_gt  = load_gt_map(FRAME, side="L", map_type="depth")
occ_mask  = load_gt_map(FRAME, side="L", map_type="occlusion")

print(f"Disparity GT shape: {disp_gt.shape}  range: [{disp_gt.min():.1f}, {disp_gt.max():.1f}] px")
print(f"Depth     GT shape: {depth_gt.shape}  range: [{depth_gt.min():.0f}, {depth_gt.max():.0f}] cm")

# === Cell 4: Visualisation – side by side ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
axes[0,0].set_title(f"Left – {ILLUM} – frame {FRAME:04d}")
axes[0,0].axis('off')

axes[0,1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
axes[0,1].set_title("Right")
axes[0,1].axis('off')

im1 = axes[1,0].imshow(disp_gt, cmap='jet', vmin=0, vmax=64)
axes[1,0].set_title("Ground-truth Disparity (pixels)")
fig.colorbar(im1, ax=axes[1,0], orientation='vertical', shrink=0.7, label='disparity (px)')

im2 = axes[1,1].imshow(depth_gt, cmap='viridis')
axes[1,1].set_title("Ground-truth Depth (cm)")
fig.colorbar(im2, ax=axes[1,1], orientation='vertical', shrink=0.7, label='depth (cm)')

plt.tight_layout()
plt.show()

# === Cell 5: Quick validity check ===
valid_pixels = np.isfinite(disp_gt) & (disp_gt > 0)
print(f"Valid disparity pixels: {valid_pixels.mean():.1%}  ({valid_pixels.sum():,} / {disp_gt.size:,})")
print(f"Approximate closest object: {depth_gt[valid_pixels].min():.1f} cm")
print(f"Approximate farthest  object: {depth_gt[valid_pixels].max():.0f} cm")