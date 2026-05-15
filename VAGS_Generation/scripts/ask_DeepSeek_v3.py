import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def setup_model_and_tokenizer():
    """
    Load the DeepSeek-V3 model and tokenizer
    """
    model_name = "deepseek-ai/DeepSeek-V3"
    
    print(f"Loading model: {model_name}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set environment variable to skip the unsupported quantization
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    
    # Initialize model with appropriate settings for inference
    # With trust_remote_code=True and force_download_config=True to handle custom configs
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to reduce memory usage
            device_map="auto",  # Automatically use GPU if available
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            # Skip quantization to avoid the fp8 error
            quantization_config=None
        )
    except ValueError as e:
        print(f"Encountered error with default loading: {e}")
        print("Trying alternative loading method...")
        
        # Try with different configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True, 
            use_cache=True,
            # Explicitly load in float16 without any quantization
            load_in_8bit=False,
            load_in_4bit=False
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """
    Generate a response to the user's question using the DeepSeek-V3 model
    """
    # Format prompt according to DeepSeek-V3's chat template
    chat = [
        {"role": "user", "content": question}
    ]
    
    # Use the tokenizer's apply_chat_template method
    try:
        # First try with the standard chat template application
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    except Exception as e:
        print(f"Chat template error: {e}")
        # Fallback to manual formatting if the template fails
        prompt = f"User: {question}\nAssistant:"
    
    print(f"Using prompt format: {prompt[:100]}...")
    
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
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    # The full text includes the prompt, so we need to remove that part
    assistant_text = generated_text[len(prompt):].strip()
    
    # If the extraction didn't work well, return everything after the last "Assistant:" if it exists
    if not assistant_text and "Assistant:" in generated_text:
        assistant_text = generated_text.split("Assistant:")[-1].strip()
    
    return assistant_text if assistant_text else generated_text

def main():
    """
    Main function to handle user interaction
    """
    parser = argparse.ArgumentParser(description="Question answering with DeepSeek-V3")
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
                print("\nDeepSeek:", response)

if __name__ == "__main__":
    main()