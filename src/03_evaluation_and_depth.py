# ────────────────────────────────────────────────────────────────
# 03_evaluation_and_depth.ipynb
# Goal: Quantitative evaluation + disparity → depth conversion
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
    evaluate_disparity,
    evaluate_depth,
    BASELINE_CM, FOCAL_PX
)

FRAME = 1
ILLUM = "fluorescent"

left, right = load_stereo_pair(illum=ILLUM, frame=FRAME)
disp_gt  = load_gt_map(FRAME, "L", "disparity")
depth_gt = load_gt_map(FRAME, "L", "depth")
occ_mask = load_gt_map(FRAME, "L", "occlusion")

# Compute
disp_sgbm = compute_disparity(left, right, mode="SGBM")
depth_sgbm = disparity_to_depth(disp_sgbm)
disp_gt = np.squeeze(disp_gt)
depth_gt = np.squeeze(depth_gt)

print(f" depth shape: {depth_sgbm.shape}   dtype: {depth_sgbm.dtype}")
print(f" depth shape: {depth_gt.shape}   dtype: {depth_gt.dtype}")

# Evaluate
metrics_disp = evaluate_disparity(disp_sgbm, disp_gt, occ_mask, nonocc_only=True)
metrics_depth = evaluate_depth(depth_sgbm, depth_gt, occ_mask)

print("=== Disparity domain metrics (non-occluded pixels) ===")
for k,v in metrics_disp.items():
    print(f"{k:12} = {v:.3f}" if isinstance(v,float) else f"{k:12} = {v}")

print("\n=== Depth domain metrics ===")
for k,v in metrics_depth.items():
    print(f"{k:12} = {v:.3f}" if isinstance(v,float) else f"{k:12} = {v}")

# === Depth visualization ===
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(depth_gt, cmap='viridis')
plt.colorbar(label='Depth (cm)')
plt.title("Ground-truth Depth")

plt.subplot(1,2,2)
plt.imshow(depth_sgbm, cmap='viridis')
plt.colorbar(label='Depth (cm)')
plt.title("Estimated Depth (SGBM)")

plt.tight_layout()
plt.show()

# === Error histogram ===
valid = np.isfinite(disp_gt) & (disp_gt > 0) & occ_mask

err = np.abs(disp_sgbm[valid] - disp_gt[valid])

plt.figure(figsize=(9,5))
sns.histplot(err, bins=80, kde=True, stat="density")
plt.title("Disparity Error Distribution – non-occluded pixels")
plt.xlabel("Absolute error (pixels)")
plt.ylabel("Density")
plt.xlim(0, 10)
plt.show()