import sys
import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

# Get the chunk file (list of CSV filenames) 
file_list_path = sys.argv[1]
with open(file_list_path) as f:
    csv_files = [line.strip() for line in f if line.strip()]

episodes_directory = '../data/episodes/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once
model_name = "ariannap22/collectiveaction_roberta_simplified_synthetic_weights"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Parameters for splitting long sentences
max_tokens = 512
stride = 256  # overlap between segments

# Process each file in the chunk
for csv_file in csv_files:
    csv_path = os.path.join(episodes_directory, csv_file)
    print(f"\nProcessing file: {csv_file}")

    try:
        df = pd.read_csv(csv_path)

        if 'collectiveAction' in df.columns:
            print(f"Skipping {csv_file}: already processed.")
            continue

        df['sentence'] = df['sentence'].fillna("").astype(str)
        predictions = []
        probabilities = []

        for sentence in df['sentence']:
            # Tokenize first without truncation to check length
            tokens = tokenizer(
                sentence,
                truncation=False,
                return_tensors="pt"
            )

            input_ids = tokens["input_ids"][0]
            n_tokens = input_ids.shape[0]

            if n_tokens <= max_tokens:
                # Short enough: single prediction
                inputs = tokenizer(
                    sentence,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = softmax(outputs.logits, dim=-1).cpu()
            else:
                # Long: split into overlapping chunks
                chunks = []
                for start in range(0, n_tokens, stride):
                    end = min(start + max_tokens, n_tokens)
                    chunk_ids = input_ids[start:end]

                    chunk = {
                        "input_ids": chunk_ids.unsqueeze(0),
                        "attention_mask": torch.ones_like(chunk_ids).unsqueeze(0)
                    }
                    chunk = {k: v.to(device) for k, v in chunk.items()}
                    chunks.append(chunk)

                # Run model on each chunk and average the softmax probabilities
                chunk_probs = []
                with torch.no_grad():
                    for chunk in chunks:
                        output = model(**chunk)
                        prob = softmax(output.logits, dim=-1).cpu()
                        chunk_probs.append(prob)

                probs = torch.stack(chunk_probs).mean(dim=0)

            pred_idx = torch.argmax(probs, dim=-1).item()
            predictions.append(pred_idx)
            probabilities.append(probs.squeeze().tolist())

        df['collectiveAction'] = predictions
        df['probCollectiveAction'] = probabilities
        df.to_csv(csv_path, index=False)
        print(f"Updated file saved: {csv_file}")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

print("\nChunk complete.")