import torch
from src.report_generation import load_gpt2_tokenizer_model, fine_tune_gpt2

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

processed_txt = "data/processed_reports.txt"
save_dir = "data/finetuned_distilgpt2-bigger"

# 1. Load reports FIRST
with open(processed_txt, 'r') as f:
    reports = [line.strip() for line in f.readlines() if line.strip()]

# fine-tuning
subset_size = 1000
reports_small = reports[:subset_size]
print(f"Using {len(reports_small)} reports for fine-tuning")

# 2. Detect device
device = get_device()
print("Using device:", device)

# 3. Load tokenizer & model ONCE
tokenizer, model = load_gpt2_tokenizer_model(device=device)
print("Model loaded")

# 4. Fine-tune once using fast settings
losses = fine_tune_gpt2(
    model,
    tokenizer,
    reports_small,
    epochs=20,       
    batch_size=2,   # safe on MPS/CPU
    device=device,
    save_path=save_dir
)

print("Training complete. Model saved.")
