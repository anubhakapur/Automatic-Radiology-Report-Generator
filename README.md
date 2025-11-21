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
