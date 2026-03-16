#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import sys
import io

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

url="https://www.gutenberg.org/files/1041/1041-0.txt"
text=requests.get(url).text

print(text[:1000])


# In[2]:


print(text[:2000])


# In[3]:


import re

lines=text.split("\n")
clean_sonnets=[]
current_sonnet=[]
roman_pattern=re.compile(r"^[IVXLCDM]+$")

for line in lines:
    stripped=line.strip()

    if roman_pattern.match(stripped):
        if len(current_sonnet)>=14:
            clean_sonnets.append("\n".join(current_sonnet[:14]))
        current_sonnet=[]
    else:
        if stripped:
            current_sonnet.append(stripped)
if len(current_sonnet)>=14:
    clean_sonnets.append("\n".join(current_sonnet[:14]))

print("Total sonnets: ", len(clean_sonnets))

if clean_sonnets:
    print("\nFirst sonnet:\n")
    print(clean_sonnets[0])


# In[4]:


for i in range(5):
    print(f"\n\n--- Sonnet {i+1} ---\n")
    print(clean_sonnets[i])


# In[5]:


def detect_theme(text):
    text_lower=text.lower()

    theme_keywords={
    "love": [
        "love", "loving", "beloved", "heart", "sweet",
        "affection", "desire", "passion", "devotion",
        "adoration", "cherish"
    ],

    "jealousy": [
        "jealous", "envy", "envious", "green",
        "suspicion", "doubt", "rival", "possess",
        "mistrust", "faithless", "betray",
        "treachery", "false", "falsehood"
    ],

    "time": [
        "time", "age", "ages", "hour", "hours",
        "season", "winter", "summer", "spring",
        "years", "eternity", "moment", "decay",
        "wither", "fade"
    ],

    "mortality": [
        "death", "die", "dying", "grave",
        "dust", "mortal", "corpse", "tomb",
        "perish", "fade", "decay",
        "ashes", "funeral"
    ],

    "betrayal": [
        "betray", "betrayed", "treason",
        "false", "falsehood", "lie", "lying",
        "faithless", "deceive", "deception",
        "treachery", "broken vow"
    ],

    "beauty": [
        "beauty", "fair", "fairness",
        "rose", "bloom", "lovely",
        "radiant", "grace", "bright",
        "golden", "divine"
    ],

    "aging": [
        "old", "wrinkle", "wrinkled",
        "winter", "decline", "wither",
        "gray", "ancient", "fading"
    ],

    "nature": [
        "sun", "moon", "stars",
        "spring", "summer", "winter",
        "rose", "flower", "garden",
        "sky", "earth", "wind",
        "storm", "sea"
    ]
}
    theme_scores={}
    for theme, keywords in theme_keywords.items():
        score=sum(text_lower.count(word) for word in keywords)
        theme_scores[theme]=score
    best_theme=max(theme_scores, key=theme_scores.get)

    if theme_scores[best_theme]==0:
        return "life"
    return best_theme




# In[6]:


#def number_lines(sonnet):
    #lines=[l.strip() for l in sonnet.split("\n") if l.strip()]
    #lines=lines[:14] 
    #numbered=[f"{i+1}.{line}" for i, line in enumerate(lines)]
    #return "\n".join(numbered)


# In[7]:


chat_dataset = []

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

for sonnet in clean_sonnets:
    theme=detect_theme(sonnet)

    example = {
        "messages": [ #chat format
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(theme)},
            {"role": "assistant", "content": sonnet}
        ]
    }

    chat_dataset.append(example)

print("Total examples:", len(chat_dataset))
print(chat_dataset[0])


# In[8]:


import json

with open("shakespeare_chat_dataset.jsonl", "w", encoding="utf-8") as f:
    for example in chat_dataset:
        f.write(json.dumps(example, ensure_ascii=False)+"\n")

print("Saved successfully.")


# In[9]:


# get_ipython().system('head shakespeare_chat_dataset.jsonl')


# In[10]:


# get_ipython().system('pip install -q transformers peft bitsandbytes accelerate trl datasets')


# In[11]:


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
tokenizer=AutoTokenizer.from_pretrained(model_name)


model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
   device_map="auto"
)


# In[12]:


import torch
print(torch.cuda.is_available())


# In[13]:


from peft import LoraConfig, get_peft_model
lora_config=LoraConfig(
    r=16,               
    lora_alpha=32,           
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model=get_peft_model(model, lora_config)
model.print_trainable_parameters()


# In[14]:


print(type(model))


# In[15]:


from datasets import load_dataset
dataset=load_dataset("json", data_files="shakespeare_chat_dataset.jsonl")
def convert_to_text(example):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return example

dataset = dataset.map(convert_to_text)
print(dataset)


# In[16]:


from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments


# In[17]:


sft_config = SFTConfig(
    output_dir="./shakespeare_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=2e-4,
    dataset_text_field="text",
    logging_steps=10,
    save_strategy="epoch",
    
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=sft_config,
)
trainer.train()

model.save_pretrained("./shakespeare_lora_adapter")



# In[18]:


import trl
print(trl.__version__)


# In[19]:


messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_user_prompt(theme)}
]
prompt=tokenizer.apply_chat_template(messages, tokenize=False)
inputs=tokenizer(prompt, return_tensors="pt").to(model.device)

outputs=model.generate(
    **inputs,
    max_new_tokens=220,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# In[20]:


base_model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_user_prompt(theme)}
]

prompt=tokenizer.apply_chat_template(messages, tokenize=False)

inputs=tokenizer(prompt, return_tensors="pt").to(base_model.device)

outputs = base_model.generate(
    **inputs,
    max_new_tokens=220,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# In[21]:


def generate_sonnet(model, theme):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(theme)}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[22]:


def line_count_score(text):
    lines=[l for l in text.split("\n") if l.strip()]
    return len(lines)


# In[23]:


theme_keywords={
    "love": [
        "love", "loving", "beloved", "heart", "sweet",
        "affection", "desire", "passion", "devotion",
        "adoration", "cherish"
    ],

    "jealousy": [
        "jealous", "envy", "envious", "green",
        "suspicion", "doubt", "rival", "possess",
        "mistrust", "faithless", "betray",
        "treachery", "false", "falsehood"
    ],

    "time": [
        "time", "age", "ages", "hour", "hours",
        "season", "winter", "summer", "spring",
        "years", "eternity", "moment", "decay",
        "wither", "fade"
    ],

    "mortality": [
        "death", "die", "dying", "grave",
        "dust", "mortal", "corpse", "tomb",
        "perish", "fade", "decay",
        "ashes", "funeral"
    ],

    "betrayal": [
        "betray", "betrayed", "treason",
        "false", "falsehood", "lie", "lying",
        "faithless", "deceive", "deception",
        "treachery", "broken vow"
    ]
    }

  




# In[24]:


def theme_score(text, theme):
    text=text.lower()
    return sum(text.count(word) for word in theme_keywords[theme])


# In[25]:


def rhyme_score(text):
    lines = [l.strip() for l in text.split('\n') if l.strip() and l[0].isdigit()]
    if len(lines) != 14:
        return 0
    
    def get_rhyme_word(line):
        parts = line.split('.', 1)
        if len(parts) < 2:
            return ""
        word = parts[1].strip().split()[-1].rstrip('.,;:?!').lower()
        return word
    
    ends = [get_rhyme_word(l) for l in lines]
    if not all(ends):
        return 0
    
    pairs = [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]
    score = 0
    for a, b in pairs:
        wa, wb = ends[a], ends[b]
        if wa and wb and (wa[-3:] == wb[-3:] or wa[-2:] == wb[-2:]):
            score += 1
    return score / 7.0


# In[26]:


archaic_words=["thou","thy","thee","doth","hath","lov'st"]
def style_score(text):
    text=text.lower()
    return sum(text.count(word) for word in archaic_words)


# In[27]:


theme="jealousy"

before_text=generate_sonnet(base_model,theme)

with open("before.txt","w", encoding="utf-8") as f:
    f.write(before_text)
    


# In[28]:


after_text=generate_sonnet(model, theme)

with open("after.txt", "w", encoding="utf-8") as f:
    f.write(after_text)


# In[29]:


with open("before.txt", "r", encoding="utf-8") as f:
    before_text = f.read()

with open("after.txt", "r", encoding="utf-8") as f:
    after_text = f.read()

print("===== BEFORE LORA =====")
print(before_text)

print("\n===== AFTER LORA =====")
print(after_text)

print("\n===== METRIC COMPARISON =====")

print("\nLine Count:")
print("Before:", line_count_score(before_text))
print("After :", line_count_score(after_text))

print("\nTheme Score:")
print("Before:", theme_score(before_text, theme))
print("After :", theme_score(after_text, theme))

print("\nStyle Score:")
print("Before:", style_score(before_text))
print("After :", style_score(after_text))

print("\nRhyme Score:")
print("Before:", rhyme_score(before_text))
print("After :", rhyme_score(after_text))


# In[30]:


test_themes = ["love", "jealousy", "time", "betrayal", "mortality"]

results = []

for theme in test_themes:
    print(f"\nGenerating for theme: {theme}")

    before_text = generate_sonnet(base_model, theme)
    after_text = generate_sonnet(model, theme)

    result = {
        "theme": theme,
        "before_line_count": line_count_score(before_text),
        "after_line_count": line_count_score(after_text),
        "before_theme_score": theme_score(before_text, theme),
        "after_theme_score": theme_score(after_text, theme),
        "before_style_score": style_score(before_text),
        "after_style_score": style_score(after_text),
        "before_rhyme_score": rhyme_score(before_text),
        "after_rhyme_score": rhyme_score(after_text)
    }

    results.append(result)


# In[31]:


print("\n================ FINAL COMPARISON TABLE ================\n")

for r in results:
    print(f"Theme: {r['theme']}")
    print("-----------------------------------------")
    print(f"Line Count       | Before: {r['before_line_count']} | After: {r['after_line_count']}")
    print(f"Theme Score      | Before: {r['before_theme_score']} | After: {r['after_theme_score']}")
    print(f"Style Score      | Before: {r['before_style_score']} | After: {r['after_style_score']}")
    print(f"Rhyme Score      | Before: {r['before_rhyme_score']} | After: {r['after_rhyme_score']}")
    print("\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




