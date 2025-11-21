from src.feature_extraction import get_vit_model, extract_features
from src.data_preprocessing import get_image_paths
import pandas as pd
import os
# Paths
merged_csv = "data/merged_dataset.csv"
features_path = "data/chest_xray_features.npy"

df = pd.read_csv(merged_csv)
df = pd.read_csv("data/merged_dataset.csv")

# 🔥 Force-correct all image paths
df['image_path'] = df['filename'].apply(
    lambda fn: os.path.join("data/images/images_normalized", fn)
)

# 🔥 Keep only files that actually exist
df = df[df['image_path'].apply(os.path.exists)]

# 🔥 Save back
df.to_csv("data/merged_dataset.csv", index=False)

print("Fixed image paths!")

image_paths = get_image_paths(df)
model = get_vit_model(device='cpu')
features = extract_features(
    model, image_paths, device='cpu', batch_size=16, save_path=features_path)
print("Features extracted and saved.")
