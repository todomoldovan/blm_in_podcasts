import os
import sys
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login

os.environ["TRANSFORMERS_CACHE"] = "/mimer/NOBACKUP/groups/naiss2024-22-185/cache"
print("Hugging Face cache set to:", os.getenv("TRANSFORMERS_CACHE"))

#access_token = ""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading the model and tokenizer...")
model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=os.environ["TRANSFORMERS_CACHE"],
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
tokenizer.pad_token_id = tokenizer.eos_token_id
print("Model and tokenizer loaded.")

classifier = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    temperature=0.01,
)
print("Classifier pipeline created.")

def generate_test_prompt6(text):
    return f"""
Classify whether the text mentions race, ethnicity, systematic racism, racial justice, or police brutality in any way ("1") or not ("0").

Return only "1" or "0" with no explanation.

text: {text}
label: """.strip()

if len(sys.argv) < 2:
    print("Usage: python process_chunk_llama.py <chunk_index>")
    sys.exit(1)

chunk_index = int(sys.argv[1])
chunk_size = 100
episodes_directory = '../data/episodes/'

all_files = sorted([f for f in os.listdir(episodes_directory) if f.endswith('.csv')])
start_idx = chunk_index * chunk_size
end_idx = min(start_idx + chunk_size, len(all_files))
chunk_files = all_files[start_idx:end_idx]

if not chunk_files:
    print(f"No files found for chunk {chunk_index}")
    sys.exit(0)

print(f"Processing chunk {chunk_index}: files {start_idx} to {end_idx - 1}")
for f in chunk_files:
    print(f" - {f}")

for csv_file in chunk_files:
    csv_path = os.path.join(episodes_directory, csv_file)
    try:
        df = pd.read_csv(csv_path)

        # if 'race' in df.columns:
        #     print(f"Skipping {csv_file}: already processed.")
        #     continue

        df['sentence'] = df['sentence'].fillna("").astype(str)

        # Here we apply the race classifier only to collective action row
        # mask = df['collectiveAction'] == 0
        # texts_to_classify = df.loc[mask, 'sentence'].tolist()

        # Here we apply the race classifier to the rows surrounding collective action 
        # Identify rows where collectiveAction == 0
        zero_idxs = df.index[df['collectiveAction'] == 0].tolist()
        # Build a set of valid neighbors (before and after)
        neighbor_idxs = set()
        for idx in zero_idxs:
            if idx > 0 and df.at[idx - 1, 'collectiveAction'] != 0: 
                neighbor_idxs.add(idx - 1)
            if idx < len(df) - 1 and df.at[idx + 1, 'collectiveAction'] != 0:
                neighbor_idxs.add(idx + 1)
        neighbor_idxs = sorted(neighbor_idxs)
        texts_to_classify = df.loc[neighbor_idxs, 'sentence'].tolist()

        predictions = []
        for text in texts_to_classify:
            prompt = generate_test_prompt6(text)
            output = classifier(prompt)[0]['generated_text'].strip()

            # Extract 0 or 1 from model output
            match = re.search(r'label:\s*(\d+)', output)
            label = match.group(1) if match else "error"
            predictions.append(label)

        # Here we apply the race classifier only to collective action row
        # df['race'] = None
        # df.loc[mask, 'race'] = predictions

        # Here we apply the race classifier to the rows surrounding collective action 
        #df['race'] = None
        for idx, label in zip(neighbor_idxs, predictions):
            if pd.isna(df.at[idx, 'race']):
                df.at[idx, 'race'] = label
            #df.at[idx, 'race'] = label

        df.to_csv(csv_path, index=False)
        print(f"Updated file saved: {csv_file}")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
