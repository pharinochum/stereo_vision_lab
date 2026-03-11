# ────────────────────────────────────────────────────────────────
# 04_illumination_analysis.ipynb
# Goal: Compare performance across lighting conditions
# ────────────────────────────────────────────────────────────────

import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


illuminations = ["daylight", "fluorescent", "lamps", "flashlight"]
algorithms    = ["BM", "SGBM"]
n_frames      = 50   # adjust according to your time / computer

results = []

for illum in illuminations:
    print(f"Processing {illum} ...")
    for algo in algorithms:
        rmse_list, bad3_list = [], []
        
        for frame in tqdm(range(0, 1800, 1800//n_frames), desc=illum+" "+algo, leave=False):
            try:
                left, right = load_stereo_pair(illum, frame)
                disp_est = compute_disparity(left, right, mode=algo)
                disp_gt  = load_gt_map(frame, "L", "disparity")
                occ      = load_gt_map(frame, "L", "occlusion")
                
                met = evaluate_disparity(disp_est, disp_gt, occ, nonocc_only=True)
                rmse_list.append(met["rmse"])
                bad3_list.append(met["bad3"])
            except Exception as e:
                continue  # skip broken frames
        
        if len(rmse_list) > 0:
            results.append({
                "illumination": illum,
                "algorithm": algo,
                "rmse_mean": np.nanmean(rmse_list),
                "bad3_mean": np.nanmean(bad3_list),
                "n_frames_valid": len(rmse_list)
            })

# === Create and show summary table ===
df = pd.DataFrame(results)
print(df.round(3))

# === Bar plot ===
#plt.figure(figsize=(10,6))
#sns.barplot(data=df, x="illumination", y="rmse_mean", hue="algorithm")
#plt.title("Average RMSE (non-occluded) – different illuminations")
#plt.ylabel("RMSE (pixels)")
#plt.show()

print(results)