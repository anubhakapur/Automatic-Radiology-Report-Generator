import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.feature_extraction import (
    get_vit_model,
    extract_features_single
)
from src.integration import find_most_similar_image
from src.report_generation import (
    load_gpt2_tokenizer_model,
    generate_report,
    clean_generated_report
)

# ----------------------------
# Config
# ----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

FEATURES_PATH = "data/chest_xray_features.npy"
MERGED_CSV = "data/merged_dataset.csv"
MODEL_DIR = "data/finetuned_distilgpt2-bigger"

features = np.load(FEATURES_PATH)
df = pd.read_csv(MERGED_CSV)

tokenizer, model = load_gpt2_tokenizer_model(MODEL_DIR, device=device)
vit = get_vit_model(device)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Automatic Radiology Report Generator", layout="wide")
st.title("Automatic Radiology Report Generator")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=350)

    with st.spinner("Extracting image features..."):
        extracted_flat = extract_features_single(vit, image, device=device)

    with st.spinner("Finding similar X-ray..."):
        idx, sim = find_most_similar_image(
            extracted_flat,
            features.reshape(features.shape[0], -1)
        )

    row = df.iloc[idx]
    raw_findings = row["MeSH"] if "MeSH" in row else row["findings"]
    clean_findings = raw_findings.replace(";", ", ").replace("/", " ")

    prompt = (
    f"Findings: {clean_findings}.\n"
    # "Impression: The radiograph shows clear evidence of chronic pulmonary hyperinflation and structural changes. "
    # "Overall, the appearance is consistent with advanced emphysematous disease. "
    "In more detail, "
    )


    st.write(f"**Matched Case ID:** {idx} (Similarity: {sim:.4f})")

    with st.spinner("Generating report..."):
        raw = generate_report(tokenizer, model, prompt, device=device)
        final = clean_generated_report(raw)

    st.subheader("📄 Generated Report")
    st.write(final)
