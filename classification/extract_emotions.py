import os
import sys
import pandas as pd
from transformers import pipeline

if len(sys.argv) < 2:
    print("Usage: python process_chunk_goemotions.py <chunk_index>")
    sys.exit(1)

chunk_index = int(sys.argv[1])
chunk_size = 1000
episodes_directory = "../data/episodes"

all_files = sorted([f for f in os.listdir(episodes_directory) if f.endswith(".csv")])
start_idx = chunk_index * chunk_size
end_idx = min(start_idx + chunk_size, len(all_files))
chunk_files = all_files[start_idx:end_idx]

if not chunk_files:
    print(f"No files found for chunk {chunk_index}")
    sys.exit(0)

print(f"Processing chunk {chunk_index}: files {start_idx} to {end_idx - 1}")

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def classify_texts(df):
    df["sentence"] = df["sentence"].fillna("").astype(str)
    max_length = 512

    results = []
    for text in df["sentence"]:
        try:
            text = text[:max_length]
            classification = classifier(text)[0]
            scores = {res["label"]: res["score"] for res in classification}
            results.append(scores)
        except Exception as e:
            print(f"Error classifying text: {text[:50]}... – {e}")
            results.append({})  # Fallback to empty row

    scores_df = pd.DataFrame(results)
    df_with_scores = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    return df_with_scores

for filename in chunk_files:
    csv_path = os.path.join(episodes_directory, filename)
    try:
        df = pd.read_csv(csv_path)
	# Skip if emotion columns already exists (e.g., gratitude)
        if "gratitude" in df.columns:
            print(f"Skipped (already has 'gratitude'): {filename}")
            continue
        df_with_scores = classify_texts(df)
        df_with_scores.to_csv(csv_path, index=False)
        print(f"Processed and saved: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nChunk complete.")
