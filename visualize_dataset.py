import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Let's load the first sample from val.csv
s1_path = Path("data/patches/sentinel1/patch_00004.npy")
s2_path = Path("data/patches/sentinel2/patch_00004.npy")
lbl_path = Path("data/patches/labels/patch_00004.npy")

s1 = np.load(s1_path)
s2 = np.load(s2_path)
lbl = np.load(lbl_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Sentinel-1 (using first channel as grayscale)
s1_vis = s1[0]
s1_vis = (s1_vis - s1_vis.min()) / (s1_vis.max() - s1_vis.min() + 1e-8)
axes[0].imshow(s1_vis, cmap='gray')
axes[0].set_title('Sentinel-1 (Channel 1)')
axes[0].axis('off')

# Plot Sentinel-2 (using first 3 channels for pseudo-RGB)
# Normalize to 0-1 for visualization
s2_vis = s2[:3].transpose(1, 2, 0)
# simple percentile clipping for better visualization
p2, p98 = np.percentile(s2_vis, (2, 98))
s2_vis = np.clip(s2_vis, p2, p98)
s2_vis = (s2_vis - p2) / (p98 - p2 + 1e-8)
axes[1].imshow(s2_vis)
axes[1].set_title('Sentinel-2 (Pseudo-RGB)')
axes[1].axis('off')

# Plot Label
axes[2].imshow(lbl, cmap='tab20')
axes[2].set_title('Label Mask')
axes[2].axis('off')

plt.tight_layout()
out_dir = Path("outputs") / "visualizations"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "dataset_sample.png"
plt.savefig(out_path)
print(f"Saved visualization to {out_path}")
