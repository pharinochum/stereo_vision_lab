import numpy as np
from justpfm import justpfm
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Dataset constants
BASELINE_CM = 10.0
FOCAL_PX    = 615.0
IMG_HEIGHT, IMG_WIDTH = 480, 640

def load_stereo_pair(illum="daylight", frame=0):
    base = f"/home/pharino/ECAM-MV/stereo_vision_lab/data/NewTsukubaStereoDataset/illumination/{illum}"
    left  = cv2.imread(f"{base}/L_{frame:05d}.png", cv2.IMREAD_COLOR) # file name L_00001.png, R_00001
    right = cv2.imread(f"{base}/R_{frame:05d}.png", cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing stereo pair at frame {frame} ({illum})")
    return left, right

def load_gt_map(frame=0, side="L", map_type="disparity"):
    base = f"/home/pharino/ECAM-MV/stereo_vision_lab/data/NewTsukubaStereoDataset/groundtruth/{map_type}_maps"
    if map_type == "disparity":
        path = f"{base}/{side}_{frame:05d}.pfm"
    elif map_type == "depth":
        path = f"{base}/{side}_{frame:05d}.pfm"
    elif map_type in ["occlusion", "discontinuity"]:
        path = f"{base}/{side}_{frame:05d}.png" if map_type == "occlusion" else f"{base}/{side}_{frame:05d}.png"
    else:
        raise ValueError("Unknown map_type")
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {map_type} map: {path}\n→ Adjust prefix in code if needed")

    if path.endswith(".pfm"):
        data = justpfm.read_pfm(path)           # ← no unpacking
        return data.astype(np.float32)
    else:  # PNG mask
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read PNG mask: {path}")
        return mask > 128   # True = valid/visible/non-occluded
 
def disparity_to_depth(disp):
    depth = np.full_like(disp, np.nan, dtype=np.float32)
    valid = (disp > 0.1) & np.isfinite(disp)
    depth[valid] = (BASELINE_CM * FOCAL_PX) / disp[valid]
    return depth


def compute_disparity(left_img, right_img, mode="SGBM"):
    left_gray  = cv2.cvtColor(left_img,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    if mode.upper() == "BM":
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    else:  # SGBM (better quality)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8*3*5**2,
            P2=32*3*5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    disp_raw = stereo.compute(left_gray, right_gray)
    disp = disp_raw.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan  # mark invalid
    return disp


def evaluate_disparity(disp_est, disp_gt, occ_mask=None, nonocc_only=True,
                       bad_thresholds=(1.0, 3.0)):
    valid = np.isfinite(disp_gt) & (disp_gt > 0)
    if occ_mask is not None and nonocc_only:
        valid &= occ_mask
    
    if valid.sum() < 100:
        return {k: np.nan for k in ["rmse","mae","bad1","bad3","valid_pct"]}
    
    err = np.abs(disp_est[valid] - disp_gt[valid])
    return {
        "rmse":     np.sqrt(np.mean(err**2)),
        "mae":      np.mean(err),
        "bad1":     np.mean(err > bad_thresholds[0]) * 100,
        "bad3":     np.mean(err > bad_thresholds[1]) * 100,
        "valid_pct": valid.mean() * 100
    }


def evaluate_depth(depth_est, depth_gt, occ_mask=None):
    valid = np.isfinite(depth_gt) & (depth_gt > 0) & np.isfinite(depth_est)
    if occ_mask is not None:
        valid &= occ_mask
    if valid.sum() < 100:
        return {k: np.nan for k in ["rmse_depth","mae_depth","bad5rel"]}
    err_abs = np.abs(depth_est[valid] - depth_gt[valid])
    err_rel = err_abs / depth_gt[valid]
    return {
        "rmse_depth": np.sqrt(np.mean(err_abs**2)),
        "mae_depth":  np.mean(err_abs),
        "bad5rel":    np.mean(err_rel > 0.05) * 100   # 5% relative error
    }
