# python scripts/read_perf_results_caption.py /path/to/folder
# python scripts/read_perf_results_caption.py 'output_iccv/coco17_no_reweight/checkpoint-26000'
# python scripts/read_perf_results_caption.py 'output_iccv/coco17_lognorm/checkpoint-26000'
# python scripts/read_perf_results_caption.py 'output_iccv/coco17_modesample/checkpoint-26000'
# python scripts/read_perf_results_caption.py 'output_iccv/coco17_cosmap/checkpoint-26000'
# python scripts/read_perf_results_caption.py 'output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000'
import json
import os

def process_metrics(input_folder, results_list=['coco17_caption_results.json', 'coco17_clair_results.json']):
    # Path to the JSON files
    caption_file = os.path.join(input_folder, results_list[0])
    clair_file = os.path.join(input_folder, results_list[1])
    
    # List to store all metrics in order
    metrics_list = []
    
    # Read caption metrics from the first file
    try:
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
            
        # Extract metrics in the specified order
        metrics_list.append(round(caption_data["metrics"]["BLEU-1"], 2))
        metrics_list.append(round(caption_data["metrics"]["BLEU-2"], 2))
        metrics_list.append(round(caption_data["metrics"]["BLEU-3"], 2))
        metrics_list.append(round(caption_data["metrics"]["BLEU-4"], 2))
        metrics_list.append(round(caption_data["metrics"]["METEOR"], 2))
        metrics_list.append(round(caption_data["metrics"]["ROUGE-1"], 2))
        metrics_list.append(round(caption_data["metrics"]["ROUGE-2"], 2))
        metrics_list.append(round(caption_data["metrics"]["ROUGE-L"], 2))
        
        print("Successfully processed caption metrics")
    except Exception as e:
        print(f"Error reading caption file: {e}")
    
    # Read clair metrics from the second file
    try:
        with open(clair_file, 'r') as f:
            clair_data = json.load(f)
        
        # Compute average score
        total_score = clair_data["summary"]["total_score"]
        count = clair_data["summary"]["count"]
        avg_score = total_score / count
        
        # Add to the metrics list
        metrics_list.append(round(avg_score,2))
        
        print("Successfully processed clair metrics")
    except Exception as e:
        print(f"Error reading clair file: {e}")
    
    print("All metrics:", metrics_list)
    return metrics_list

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = "."  # Default to current directory

    results_list=['coco17_caption_results.json', 'coco17_clair_results.json']
    results_list=['coco14_caption_results.json', 'coco14_clair_results.json']
    
    results = process_metrics(input_folder, results_list)

    output_str = ' & '.join([str(val) for val in results])
    print(output_str)