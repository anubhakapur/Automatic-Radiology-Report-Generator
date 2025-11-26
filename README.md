Automated radiology report generation is an important challenge in medical imaging, requiring models that can both understand visual patterns and communicate clinical findings in clear, structured language. While existing deep learning systems can classify abnormalities or retrieve similar cases, they often fall short in producing coherent, contextually accurate, and clinically meaningful full-length reports. Vision-only models struggle to translate pixel information into higher-level reasoning, and language models alone cannot generate trustworthy descriptions without medically relevant conditioning.
This project presents a hybrid, retrieval-augmented generative AI framework that combines visual encoding, clinical similarity search, and language modeling to generate radiologist-style chest X-ray reports. The system performs:
Image feature extraction using a Vision Transformer (vit_base_patch16_224)
Nearest-neighbor retrieval of similar radiographs via cosine similarity
Prompt generation using MeSH terms and findings from retrieved cases
Report synthesis using a fine-tuned DistilGPT-2 model
Performance evaluation through BLEU, ROUGE, BERTScore, perplexity, and embedding visualizations
The workflow is trained and validated on the Indiana University Chest X-ray dataset, enabling the model to learn realistic clinical writing patterns from paired images and expert-authored reports. Unlike conventional classification pipelines, this approach treats report creation as an end-to-end retrieval-augmented narrative generation task, supporting more clinically aligned and context-aware outputs.
The system integrates:
A ViT-based visual encoder producing 768-dimensional embeddings
A retrieval module for identifying top-K=1 similar case
A prompt engineering pipeline based on MeSH-derived cues
A fine-tuned DistilGPT-2 generator for producing full diagnostic reports
A complete evaluation suite and a user-friendly inference interface

Dataset:
Source: Indiana University Chest X-ray OpenI
Included: PNG images, indiana_reports.csv, indiana_projections.csv
Data Path: Place all data inside /data/ (see below).
Privacy: Fully anonymized, public research dataset.

Steps to run:
1)git clone https://github.com/anubhakpur/Automatic-Radiology-Report-Generator.git
cd RadiologyReportGen-AI
pip install -r requirements.txt

2)python scripts/preprocess_data.py
3)python scripts/extract_features.py
4)python scripts/train_gpt2.py
5)python scripts/generate_report.py
6)python scripts/evaluate.py

Project Structure:
RadiologyReportGen-AI/
├── data/                  # Place your dataset files and images here
│   ├── indiana_reports.csv
│   ├── indiana_projections.csv
│   └── images/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── report_generation.py
│   ├── integration.py
│   ├── evaluation.py
│   └── utils.py
├── scripts/
│   ├── preprocess_data.py
│   ├── extract_features.py
│   ├── train_gpt2.py
│   ├── generate_report.py
│   └── evaluate.py
├── requirements.txt
├── README.md
