import json
import requests
import time
import argparse
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-local storage for rate limiting
thread_local = threading.local()

def load_json_data(file_path):
    """Load caption data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_caption_pair(api_key, candidate_caption, reference_caption, model="gpt-3.5-turbo"):
    """
    Send a caption pair to ChatGPT API and get similarity score
    """
    prompt = f"""You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_caption}
Reference set:
{reference_caption}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set? (dict format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that evaluates caption similarity."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        return {"score": 0, "reason": f"API error: {response.status_code}"}
    
    result = response.json()
    message_content = result["choices"][0]["message"]["content"]
    
    # Manually extract score from response
    try:
        # Try to parse JSON from the message
        score_dict = json.loads(message_content)
        return score_dict
    except json.JSONDecodeError:
        # If that fails, try to find the score in the text
        try:
            import re
            score_match = re.search(r'["\']score["\']\s*:\s*(\d+)', message_content)
            reason_match = re.search(r'["\']reason["\']\s*:\s*["\'](.*?)["\']', message_content)
            
            if score_match:
                score = int(score_match.group(1))
                reason = reason_match.group(1) if reason_match else "No reason provided"
                return {"score": score, "reason": reason}
            else:
                return {"score": 0, "reason": "Failed to parse response"}
        except Exception as e:
            return {"score": 0, "reason": "Failed to parse response"}

def process_single_caption(image_id, pair, api_key, model, delay):
    """
    Process a single caption pair with rate limiting
    """
    candidate = pair.get('generated', '')
    reference = pair.get('ground_truth', '')
    
    # Add delay for rate limiting
    if delay > 0:
        time.sleep(delay)
    
    try:
        similarity = evaluate_caption_pair(api_key, candidate, reference, model)
        
        result = {
            "image_id": image_id,
            "candidate": candidate,
            "reference": reference,
            "similarity": similarity,
            "success": True
        }
        
        return result
    except Exception as e:
        return {
            "image_id": image_id,
            "candidate": candidate,
            "reference": reference,
            "similarity": {"score": 0, "reason": f"Error: {str(e)}"},
            "success": False
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate caption similarity using ChatGPT (Parallelized)')
    parser.add_argument('--input', type=str, required=True, help='Path to the JSON file with caption pairs')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output', type=str, default='similarity_results.json', help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini', help='OpenAI model to use, gpt-5-mini|gpt-4-turbo-preview|gpt-4o')
    parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of caption pairs to evaluate')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between API calls in seconds (per worker)')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    data = load_json_data(args.input)
    
    captions = data.get("captions", {})
    print(f"Found {len(captions)} caption pairs")
    
    results = {
        "caption_evaluations": {},
        "summary": {"total_score": 0, "count": 0}
    }
    
    # Get the list of caption pairs to process
    caption_ids = list(captions.keys())
    
    # Limit the number of pairs if specified
    if args.max_pairs is not None and args.max_pairs < len(caption_ids):
        caption_ids = caption_ids[:args.max_pairs]
        print(f"Limiting evaluation to {args.max_pairs} pairs as specified.")
    
    print(f"Processing {len(caption_ids)} caption pairs with {args.workers} workers...")
    
    # Process caption pairs in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(
                process_single_caption,
                image_id,
                captions[image_id],
                args.api_key,
                args.model,
                args.delay
            ): image_id
            for image_id in caption_ids
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(caption_ids)) as pbar:
            for future in as_completed(future_to_id):
                image_id = future_to_id[future]
                try:
                    result = future.result()
                    
                    # Store result
                    results["caption_evaluations"][result["image_id"]] = {
                        "candidate": result["candidate"],
                        "reference": result["reference"],
                        "similarity": result["similarity"]
                    }
                    
                    if "score" in result["similarity"]:
                        score = result["similarity"]["score"]
                        results["summary"]["total_score"] += score
                        results["summary"]["count"] += 1
                        pbar.set_postfix({"Last Score": f"{score}/100", "Avg": f"{results['summary']['total_score']/results['summary']['count']:.1f}"})
                    
                except Exception as e:
                    print(f"\nError processing image {image_id}: {e}")
                
                pbar.update(1)
    
    # Calculate average score before saving
    if results["summary"]["count"] > 0:
        avg_score = results["summary"]["total_score"] / results["summary"]["count"]
        results["summary"]["average_score"] = avg_score

    parent_dir = os.path.dirname(args.input)
    args.output = os.path.splitext(os.path.basename(args.input))[0]
    args.output = f'{os.path.basename(args.output)}_clair.json'
    args.output = os.path.join(parent_dir, args.output)
    
    # Save the test result
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {args.output}")
    if results["summary"]["count"] > 0:
        print(f"Average CLAIR similarity score: {avg_score:.2f}")
        print(f"Evaluated {results['summary']['count']} out of {len(captions)} caption pairs successfully.")

if __name__ == "__main__":
    main()