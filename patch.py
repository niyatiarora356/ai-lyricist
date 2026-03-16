import json

path = r"c:/Users/KIIT/OneDrive/ai_lyricist/ai-lyricist-1.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        if isinstance(source, list):
            source_str = "".join(source)
            if 'model_name = "Qwen/Qwen2-1.5B-Instruct"' in source_str:
                source_str = source_str.replace('model_name = "Qwen/Qwen2-1.5B-Instruct"', 'model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"')
            if "learning_rate=2e-5," in source_str:
                source_str = source_str.replace("learning_rate=2e-5,", "learning_rate=2e-4,\n    dataset_text_field=\"text\",\n    max_seq_length=512,")
            cell["source"] = source_str.splitlines(True)
        elif isinstance(source, str):
            if 'model_name = "Qwen/Qwen2-1.5B-Instruct"' in source:
                source = source.replace('model_name = "Qwen/Qwen2-1.5B-Instruct"', 'model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"')
            if "learning_rate=2e-5," in source:
                source = source.replace("learning_rate=2e-5,", "learning_rate=2e-4,\n    dataset_text_field=\"text\",\n    max_seq_length=512,")
            cell["source"] = source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f)
print("Notebook patched successfully!")
