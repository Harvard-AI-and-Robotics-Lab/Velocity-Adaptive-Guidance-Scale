# python scripts/ask_Llama_3.2_clair.py --candidate "A dog chasing a ball" --reference "A puppy playing with a toy"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json

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

def generate_response(model, tokenizer, candidate_captions, reference_captions, max_length=1024):
    """
    Generate a response to the caption comparison question
    """
    # Format the question for the model
    question = f"""You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_captions}
Reference set:
{reference_captions}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set? (dict format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)"""
    
    # Format prompt for Llama 3.2 Instruct model
    prompt = f"<|begin_of_text|><|user|>\n{question}<|end_of_turn|>\n<|assistant|>\n"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.2,  # Lower temperature for more precise/deterministic responses
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
    
    # Try to parse the response as a JSON dict
    try:
        # Find the dictionary-like part in the response
        import re
        dict_match = re.search(r'\{.*\}', response, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            result = json.loads(dict_str)
            return result
        else:
            # If no dictionary found, return a formatted response
            return {
                "score": 0,
                "reason": "Failed to parse model response: " + response[:100] + "..."
            }
    except json.JSONDecodeError:
        # If parsing fails, return the raw response
        return {
            "score": 0,
            "reason": "Failed to parse model response: " + response[:100] + "..."
        }

def main():
    """
    Main function to handle user interaction
    """
    parser = argparse.ArgumentParser(description="Compare image captions with Llama 3.2 3B Instruct")
    parser.add_argument("--candidate", type=str, help="Candidate caption(s)")
    parser.add_argument("--reference", type=str, help="Reference caption(s)")
    args = parser.parse_args()
    
    print("Loading model, please wait...")
    model, tokenizer = setup_model_and_tokenizer()
    print("Model loaded successfully!")
    
    # Interactive mode if no captions provided
    if args.candidate and args.reference:
        response = generate_response(model, tokenizer, args.candidate, args.reference)
        print("\nCandidate captions:", args.candidate)
        print("\nReference captions:", args.reference)
        print("\nComparison result:", json.dumps(response, indent=2))
    else:
        print("\nEnter caption sets to compare (type 'exit' to quit):")
        
        while True:
            candidate_captions = input("\nCandidate captions: ")
            
            if candidate_captions.lower() in ["exit", "quit", "q"]:
                break
                
            reference_captions = input("\nReference captions: ")
            
            if reference_captions.lower() in ["exit", "quit", "q"]:
                break
                
            if candidate_captions.strip() and reference_captions.strip():
                response = generate_response(model, tokenizer, candidate_captions, reference_captions)
                print("\nComparison result:", json.dumps(response, indent=2))

if __name__ == "__main__":
    main()