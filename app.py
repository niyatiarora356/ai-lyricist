from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from contextlib import asynccontextmanager
import os

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./shakespeare_lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Starting server... Loading model on {DEVICE}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model with Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Model loaded successfully!")
    yield
    # Cleanup (if any)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class SonnetRequest(BaseModel):
    theme: str

class SonnetResponse(BaseModel):
    theme: str
    sonnet: str

SYSTEM_PROMPT = (
    "You are a Shakespearean poet."
    "You must write exactly 14 numbered lines."
    "You must strictly follow the rhyme scheme ABAB CDCD EFEF GG."
    "Ensure rhyming words appear at the end of lines."
    "Only output the 14 numbered lines."
    "Do not include explanations or headings."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Lyricist FastAPI Server!"}

@app.post("/generate", response_model=SonnetResponse)
async def generate_sonnet(request: SonnetRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    user_prompt = f"Write exactly 14 numbered lines in Shakespearean sonnet form about {request.theme}, following the rhyme scheme ABAB CDCD EFEF GG."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in full_output:
            sonnet = full_output.split("assistant")[-1].strip()
        else:
            sonnet = full_output.strip()
            
        return SonnetResponse(theme=request.theme, sonnet=sonnet)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
