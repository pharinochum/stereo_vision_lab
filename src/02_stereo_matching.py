# ────────────────────────────────────────────────────────────────
# 02_stereo_matching.ipynb
# Goal: Compute disparity maps using OpenCV StereoBM and StereoSGBM
# ────────────────────────────────────────────────────────────────
%cd ..    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    load_stereo_pair,
    load_gt_map,
    disparity_to_depth,
    compute_disparity,
    BASELINE_CM, FOCAL_PX
)

sns.set_style("whitegrid")
%matplotlib inline

# === Cell 1: Load example frame ===
left, right = load_stereo_pair(illum="daylight", frame=600)

# === Cell 2: Compute both algorithms ===
disp_bm   = compute_disparity(left, right, mode="BM")
disp_sgbm = compute_disparity(left, right, mode="SGBM")

print(f"StereoBM   disparity shape: {disp_bm.shape}   dtype: {disp_bm.dtype}")
print(f"StereoSGBM disparity shape: {disp_sgbm.shape}  dtype: {disp_sgbm.dtype}")

# === Cell 3: Visual comparison ===
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
axes[0].set_title("Left image")
axes[0].axis('off')

im1 = axes[1].imshow(disp_bm, cmap='jet', vmin=0, vmax=64)
axes[1].set_title("StereoBM")
fig.colorbar(im1, ax=axes[1], shrink=0.6)

im2 = axes[2].imshow(disp_sgbm, cmap='jet', vmin=0, vmax=64)
axes[2].set_title("StereoSGBM (recommended)")
fig.colorbar(im2, ax=axes[2], shrink=0.6)

plt.tight_layout()
plt.show()

# === Cell 4: Error map example (qualitative) ===
disp_gt = load_gt_map(600, side="L", map_type="disparity")
valid = np.isfinite(disp_gt) & (disp_gt > 0)
valid  = np.squeeze(valid)
#Remove singleton dimensions (works for both (H,W,1) and (1,H,W))
disp_gt = np.squeeze(disp_gt)
err_sgbm = np.abs(disp_sgbm - disp_gt)
err_sgbm[~valid] = np.nan

plt.figure(figsize=(10,5))
plt.imshow(err_sgbm, cmap='hot', vmin=0, vmax=5)
plt.colorbar(label='Absolute error (pixels)')
plt.title("StereoSGBM – Absolute Error Map")
plt.axis('off')
plt.show()