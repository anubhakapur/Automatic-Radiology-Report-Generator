from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def load_gpt2_tokenizer_model(model_path=None, device = "cpu"):
    """
    Loads GPT-2 tokenizer and model. If model_path is None, loads pretrained.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    if model_path:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.to(device)
    return tokenizer, model


def tokenize_reports(tokenizer, report_list, max_length=218):
    """
    Tokenizes list of reports for GPT-2.
    """
    return tokenizer(report_list, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')


def fine_tune_gpt2(model, tokenizer, reports, epochs=2, batch_size=2, lr=5e-5, device = "cpu", save_path=None):
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW

    class ReportsDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    encodings = tokenize_reports(tokenizer, reports)
    encodings['labels'] = encodings['input_ids'].clone()

    # Move everything to device upfront
    for k in encodings:
        encodings[k] = encodings[k].to(device)

    dataset = ReportsDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    losses = []
    scaler = torch.amp.GradScaler("mps")

    for epoch in range(epochs):
        print("Epoch:", epoch+1)
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            with torch.amp.autocast("mps"):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    if save_path:
        model.save_pretrained(save_path)

    return losses



def generate_report(tokenizer, model, prompt_text, max_length=218, device='cpu'):
    """
    Generates report using fine-tuned GPT-2.
    """
    # model.to(device)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=int(max_length),
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # ensures no crash
    )

    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

def clean_generated_report(report):
    report = report.replace("[IDX]", "")
    report = report.strip()
    report = " ".join(report.split())
    if report:
        report = report[0].upper() + report[1:]
    return report

