
import os
import json
from datasets import load_dataset

def main():
    print("Downloading WikiText-2-raw-v1...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter out empty lines (WikiText-2 raw contains empty strings and titles)
    # We want actual text content for KL alignment.
    # Lines that start and end with '=' are titles.
    
    print("Processing text...")
    processed_data = []
    for entry in dataset:
        text = entry["text"].strip()
        if text and not (text.startswith("=") and text.endswith("=")):
            # To keep it consistent with our QA JSONL format, 
            # we can just put it in a "text" field.
            processed_data.append({"text": text})
    
    output_path = "data/wikitext_train.jsonl"
    print(f"Saving {len(processed_data)} lines to {output_path}...")
    
    with open(output_path, "w") as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + "\n")
            
    print("WikiText-2 download and processing complete.")

if __name__ == "__main__":
    main()
