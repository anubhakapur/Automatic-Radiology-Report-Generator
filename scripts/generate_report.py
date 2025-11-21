from src.report_generation import load_gpt2_tokenizer_model, generate_report,clean_generated_report
from src.integration import find_most_similar_image
import pandas as pd
import numpy as np
import torch
def clean_prompt(prompt):
    # Remove placeholders and extra whitespace
    prompt = prompt.replace("xxxx", "")
    prompt = prompt.replace("[IDX]", "")
    prompt = prompt.strip()
    return prompt

# Paths
features_path = "data/chest_xray_features.npy"
merged_csv = "data/merged_dataset.csv"
finetuned_model_dir = "data/finetuned_distilgpt2"

# Load data
features = np.load(features_path)
df = pd.read_csv(merged_csv)
tokenizer, model = load_gpt2_tokenizer_model(
    finetuned_model_dir, device='cpu')

# Assume uploaded image features is a vector: extracted_features_flat (197*768,)
# e.g., extracted_features_flat = ... (get this by your own code)
# For demo:
# Just using first image as an example
extracted_features_flat = features[0].reshape(-1)

idx, sim_score = find_most_similar_image(
    extracted_features_flat, features.reshape(features.shape[0], -1))
row = df.iloc[idx]
prompt_text = row['MeSH'] if 'MeSH' in row else row['findings']

# device = "mps" if torch.backends.mps.is_available() else "cpu"
raw_report = generate_report(tokenizer, model, prompt_text,218, device="cpu")
report = clean_generated_report(raw_report)

print("Generated Report:\n", report)
