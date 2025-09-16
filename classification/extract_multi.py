import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

if len(sys.argv) < 2:
    print("Usage: python process_chunk.py <chunk_index>")
    sys.exit(1)

chunk_index = int(sys.argv[1])
chunk_size = 100
episodes_directory = '../data/episodes/'

all_files = sorted([f for f in os.listdir(episodes_directory) if f.endswith('.csv')])
start_idx = chunk_index * chunk_size
end_idx = min(start_idx + chunk_size, len(all_files))
chunk_files = all_files[start_idx:end_idx]

chunk_files = all_files[start_idx:end_idx]

if not chunk_files:
    print(f"No files found for chunk {chunk_index}")
    sys.exit(0)

print(f"Processing chunk {chunk_index}: files {start_idx} to {end_idx - 1}")
print("Files in this chunk:")
for f in chunk_files:
    print(f" - {f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ariannap22/collectiveaction_sft_annotated_only_v6_prompt_v6_p100_synthetic_balanced_more_layered"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

print("Model loaded.")
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

dim_def = {
    'Problem-Solution': "The comment highlights an issue and possibly suggests a way to fix it, often naming those responsible.",
    'Call-to-Action': "The comment asks readers to take part in a specific activity, effort, or movement.",
    'Intention': "The commenter shares their own desire to do something or be involved in solving a particular issue.",
    'Execution': "The commenter is describing their personal experience taking direct actions towards a common goal."
}
categories = list(dim_def.keys())

def generate_prompt(text):
    return f"""
You have the following knowledge about levels of participation in collective action that can be expressed in social media comments: {dim_def}. 

### Definitions and Criteria:
**Collective Action Problem:** A present issue caused by human actions or decisions that affects a group and can be addressed through individual or collective efforts.

**Participation in collective action**: A comment must clearly reference a collective action problem, social movement, or activism by meeting at least one of the levels in the list {categories}.

Classify the following social media comment into one of the levels within the list {categories}. 

### Example of correct output format:
text: xyz
label: None

Return the answer as the corresponding participation in collective action level label.

text: {text}
label: """.strip()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.1)

for csv_file in chunk_files:
    csv_path = os.path.join(episodes_directory, csv_file)
    try:
        df = pd.read_csv(csv_path)

        if 'collectiveActionLevel' in df.columns:
            print(f"Skipping {csv_file}: already processed.")
            continue
        if 'collectiveAction' not in df.columns:
            print(f"Skipping {csv_file}: missing collectiveAction column.")
            continue

        df['sentence'] = df['sentence'].fillna("").astype(str)
        mask = df['collectiveAction'] == 0
        texts_to_classify = df.loc[mask, 'sentence'].tolist()

        predictions = []
        for text in texts_to_classify:
            prompt = generate_prompt(text)
            output = pipe(prompt)[0]['generated_text']
            answer = output.split("label:")[-1].strip()

            for category in categories:
                if category.lower() in answer.lower():
                    predictions.append(category)
                    break
            else:
                predictions.append("error")

        df['collectiveActionLevel'] = None
        df.loc[mask, 'collectiveActionLevel'] = predictions
        df.to_csv(csv_path, index=False)
        print(f"Updated file saved: {csv_file}")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")