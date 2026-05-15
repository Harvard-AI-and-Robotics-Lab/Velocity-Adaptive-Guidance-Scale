# --------------------------------------------------------------------------
#
# A script to aggregate evaluations from multiple JSON files, group them by
# sample ID, and provide a comparative view of model performance on each sample.
#
# --------------------------------------------------------------------------

import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def calculate_sentence_metrics(generated_sentence, ground_truth_sentence):
    """
    Calculates BLEU, METEOR, and ROUGE scores for a single sentence pair.
    """
    reference = [word_tokenize(ground_truth_sentence.lower())]
    hypothesis = word_tokenize(generated_sentence.lower())
    rouge_ref_str = ground_truth_sentence.lower()
    rouge_hyp_str = generated_sentence.lower()

    metrics = {}
    smoothie = SmoothingFunction().method1

    metrics['BLEU-4'] = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) * 100
    metrics['METEOR'] = meteor_score(reference, hypothesis) * 100

    try:
        if not rouge_hyp_str.strip() or not rouge_ref_str.strip():
            metrics['ROUGE-L'] = 0.0
        else:
            rouge = Rouge()
            scores = rouge.get_scores([rouge_hyp_str], [rouge_ref_str])[0]
            metrics['ROUGE-L'] = scores['rouge-l']['f'] * 100
    except ValueError:
        metrics['ROUGE-L'] = 0.0

    return metrics

def aggregate_evaluations_by_id(file_paths):
    """
    Reads all JSON files and aggregates their evaluations, keyed by sample ID.
    """
    aggregated_results = {}
    print("🚀 Starting evaluation process...\n")

    for file_path in file_paths:
        try:
            # Use the directory name as a short-hand for the model/experiment name
            model_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            print(f"  - Reading data from: {model_name}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            captions_data = data.get("captions", {})
            if not captions_data:
                continue

            for sample_id, details in captions_data.items():
                if sample_id not in aggregated_results:
                    aggregated_results[sample_id] = {
                        "ground_truth": details.get("ground_truth", ""),
                        "evaluations": []
                    }

                generated = details.get("generated", "")
                ground_truth = details.get("ground_truth", "")

                if generated and ground_truth:
                    metrics = calculate_sentence_metrics(generated, ground_truth)
                    aggregated_results[sample_id]["evaluations"].append({
                        "source_model": model_name,
                        "generated": generated,
                        "metrics": metrics
                    })

        except Exception as e:
            print(f"  -> Could not process file {file_path}. Error: {e}")

    print("\n✅ Aggregation complete.")
    return aggregated_results

if __name__ == "__main__":
    # Define the list of your input JSON files
    input_files = [
        'output_baseline/coco17_sd35_no_reweight/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_lognorm/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_modesample/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_cosmap/checkpoint-26000/coco17_caption_results.json',
        'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/coco17_caption_results.json'
    ]
    
    output_filename = "data4paper/comparison_similarity_results.txt"

    # Aggregate all data from the files, grouped by sample ID
    aggregated_data = aggregate_evaluations_by_id(input_files)

    # --- Write all results to the output file ---
    if not aggregated_data:
        print("No data was aggregated. Please check file paths and content.")
    else:
        print(f"\n✍️ Saving all results to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("📊 Comparative Evaluation Results\n")
            f.write("="*70 + "\n\n")

            # Iterate over ALL sample IDs to write the full results
            for sample_id, data in aggregated_data.items():
                f.write(f"--- Comparison for Sample ID: {sample_id} ---\n")
                f.write(f"  📝 Ground Truth: '{data['ground_truth']}'\n\n")

                # Sort evaluations by model name for consistent ordering
                sorted_evals = sorted(data['evaluations'], key=lambda x: x['source_model'])

                for eval_item in sorted_evals:
                    metrics = eval_item['metrics']
                    f.write(f"  ▶ From Model: {eval_item['source_model']}\n")
                    f.write(f"    Generated: '{eval_item['generated']}'\n")
                    f.write(f"    Scores -> BLEU-4: {metrics['BLEU-4']:.2f}, "
                            f"METEOR: {metrics['METEOR']:.2f}, "
                            f"ROUGE-L: {metrics['ROUGE-L']:.2f}\n\n")
                
                f.write("-" * 70 + "\n\n")
        
        print(f"🎉 Success! All results have been saved to '{output_filename}'.")