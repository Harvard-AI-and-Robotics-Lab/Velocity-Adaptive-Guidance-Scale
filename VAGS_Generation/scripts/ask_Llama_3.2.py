# python scripts/ask_Llama_3.2.py --question "What is machine learning?"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def setup_model_and_tokenizer():
    """
    Load the Llama 3.2 model and tokenizer
    """
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f"Loading model: {model_name}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize model with appropriate settings for inference
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to reduce memory usage
        device_map="auto"  # Automatically use GPU if available
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """
    Generate a response to the user's question using the Llama model
    """
    # Format prompt for Llama 3.2 Instruct model
    prompt = f"<|begin_of_text|><|user|>\n{question}<|end_of_turn|>\n<|assistant|>\n"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    response_start = generated_text.find("<|assistant|>") + len("<|assistant|>")
    response_end = generated_text.find("<|end_of_turn|>", response_start)
    
    if response_end == -1:  # If end token not found, return everything after assistant tag
        response = generated_text[response_start:].strip()
    else:
        response = generated_text[response_start:response_end].strip()
    
    return response

def main():
    """
    Main function to handle user interaction
    """
    parser = argparse.ArgumentParser(description="Question answering with Llama 3.2 3B Instruct")
    parser.add_argument("--question", type=str, help="Question to ask the model")
    args = parser.parse_args()
    
    print("Loading model, please wait...")
    model, tokenizer = setup_model_and_tokenizer()
    print("Model loaded successfully!")
    
    # Interactive mode if no question provided
    if args.question:
        response = generate_response(model, tokenizer, args.question)
        print("\nQuestion:", args.question)
        print("\nResponse:", response)
    else:
        print("\nEnter your questions (type 'exit' to quit):")
        
        while True:
            question = input("\nYou: ")
            
            if question.lower() in ["exit", "quit", "q"]:
                break
                
            if question.strip():
                response = generate_response(model, tokenizer, question)
                print("\nLlama:", response)

if __name__ == "__main__":
    main()