import os
from pathlib import Path
import pandas as pd
from src.models.dataset import MultimodalSatelliteDataset

print("="*50)
print("GeoFusion AI Dataset Summary")
print("="*50)

splits = ['train', 'val', 'test']
for split in splits:
    split_file = Path(f'data/splits/{split}.csv')
    if split_file.exists():
        dataset = MultimodalSatelliteDataset(split_csv_path=split_file)
        print(f"\n[Split: {split.upper()}]")
        print(f"Total Samples: {len(dataset)}")
        if len(dataset) > 0:
            s1, s2, lbl = dataset[0]
            print(f"Sample 0 -> Sentinel-1 Shape: {s1.shape} (Channels, H, W)")
            print(f"Sample 0 -> Sentinel-2 Shape: {s2.shape} (Channels, H, W)")
            print(f"Sample 0 -> Label Mask Shape: {lbl.shape} (H, W)")
    else:
        print(f"\n[Split: {split.upper()}] - File not found")

print("\n" + "="*50)
