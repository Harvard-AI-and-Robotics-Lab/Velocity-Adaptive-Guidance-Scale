import os
import json

# input_folder = "/path/to/datasets/coco17/train"
input_folder = "/path/to/datasets/coco17/validation"

# Ensure output folder exists (optional; here we write next to the JSONs)
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)
    txt_path = os.path.join(input_folder, filename.replace(".json", ".txt"))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = data.get("captions", [])
    if not isinstance(captions, list):
        print(f"Warning: no valid captions in {filename}")
        continue

    with open(txt_path, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(caption.strip() + "\n")

    print(f"Wrote {len(captions)} captions to {txt_path}")
