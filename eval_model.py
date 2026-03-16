import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import os

# Configuration
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./shakespeare_lora_adapter"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# System Prompt
SYSTEM_PROMPT = (
    "You are a Shakespearean poet."
    "You must write exactly 14 numbered lines."
    "You must strictly follow the rhyme scheme ABAB CDCD EFEF GG."
    "Ensure rhyming words appear at the end of lines."
    "Only output the 14 numbered lines."
    "Do not include explanations or headings."
)

def build_user_prompt(theme):
    return (
        f"Write exactly 14 numbered lines in Shakespearean sonnet form about {theme}, "
        "following the rhyme scheme ABAB CDCD EFEF GG."
    )

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load Model with Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def generate_sonnet(target_model, theme):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(theme)}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and remove the prompt
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part (after the last user prompt)
    # TinyLlama chat template usually results in something like:
    # <|system|>...<|user|>...<|assistant|>...
    # The tokenizer.decode with skip_special_tokens might vary.
    # Let's try to isolate the assistant response.
    if "assistant" in full_output:
        response = full_output.split("assistant")[-1].strip()
    else:
        response = full_output.strip()
    
    return response

# Metrics
def line_count_score(text):
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines)

archaic_words = ["thou", "thy", "thee", "doth", "hath", "lov'st", "art", "shall", "wilt"]
def style_score(text):
    text = text.lower()
    return sum(text.count(word) for word in archaic_words)

def rhyme_score(text):
    lines = [l.strip() for l in text.split('\n') if l.strip() and (l[0].isdigit() or any(c.isalpha() for c in l))]
    # Filter to get exactly 14 if possible, or up to 14
    lines = lines[:14]
    if len(lines) < 14: return 0
    
    def get_rhyme_word(line):
        # Remove numbers if present at start
        clean_line = line.lstrip('0123456789. ')
        words = clean_line.strip().split()
        if not words: return ""
        word = words[-1].rstrip('.,;:?!').lower()
        return word
    
    ends = [get_rhyme_word(l) for l in lines]
    if not all(ends): return 0
    
    pairs = [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]
    score = 0
    for a, b in pairs:
        wa, wb = ends[a], ends[b]
        if wa and wb and (wa[-3:] == wb[-3:] or wa[-2:] == wb[-2:] or wa[-1] == wb[-1]):
            # Very loose rhyme check for evaluation
            score += 1
    return score / 7.0

# Evaluation
test_themes = ["love", "time", "nature"]
print("\n--- EVALUATION ---")

for theme in test_themes:
    print(f"\nTheme: {theme}")
    
    print("Generating with Base Model...")
    base_sonnet = generate_sonnet(base_model, theme)
    
    print("Generating with Fine-Tuned Model...")
    ft_sonnet = generate_sonnet(model, theme)
    
    print(f"\n[BASE MODEL OUTPUT]\n{base_sonnet}")
    print(f"\n[FINE-TUNED MODEL OUTPUT]\n{ft_sonnet}")
    
    print("\nMetrics (Base vs FT):")
    print(f"Line Count: {line_count_score(base_sonnet)} vs {line_count_score(ft_sonnet)}")
    print(f"Style Score: {style_score(base_sonnet)} vs {style_score(ft_sonnet)}")
    print(f"Rhyme Score: {rhyme_score(base_sonnet):.2f} vs {rhyme_score(ft_sonnet):.2f}")
    print("-" * 30)

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation Results\n")
    f.write("==================\n")
    # In a real run we'd log more details
