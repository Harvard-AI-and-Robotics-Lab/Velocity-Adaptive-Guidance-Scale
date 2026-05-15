import os
import json
from typing import List, Dict, Tuple

def extract_metrics_from_folder_list(folder_list: List[str]) -> Tuple[List[float], List[float]]:
    """
    Extract METEOR and ROUGE-1 metrics from coco17_caption_results.json files in the specified folders.
    
    Args:
        folder_list (List[str]): List of folder paths to search in
        
    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - List of METEOR scores
            - List of ROUGE-1 scores
    """
    meteor_scores = []
    rouge1_scores = []
    
    for folder_path in folder_list:
        json_file_path = os.path.join(folder_path, "coco17_caption_results.json")
        
        # Check if the JSON file exists in this folder
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                
                # Extract metrics
                if "metrics" in data:
                    meteor_score = data["metrics"].get("METEOR")
                    rouge1_score = data["metrics"].get("ROUGE-1")
                    
                    if meteor_score is not None:
                        meteor_scores.append(meteor_score)
                    
                    if rouge1_score is not None:
                        rouge1_scores.append(rouge1_score)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading file {json_file_path}: {e}")
    
    return meteor_scores, rouge1_scores


def save_metrics_to_file(folder_list: List[str], meteor_scores: List[float], rouge1_scores: List[float], output_file: str) -> None:
    """
    Save the extracted metrics to a file.
    
    Args:
        folder_list (List[str]): List of folder paths
        meteor_scores (List[float]): List of METEOR scores
        rouge1_scores (List[float]): List of ROUGE-1 scores
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as file:
        file.write("Folder,METEOR,ROUGE-1\n")
        
        for i in range(len(meteor_scores)):
            folder_name = os.path.basename(folder_list[i])
            file.write(f"{folder_name},{meteor_scores[i]},{rouge1_scores[i]}\n")


def main():
    # Get the folder list input
    # folder_input = input("Enter the folders (comma-separated): ")
    # folder_list = [folder.strip() for folder in folder_input.split(',')]

    folder_list = ['output_iccv/coco17_proposed_lambda_curva_0/checkpoint-26000/',
                'output_iccv/coco17_proposed_lambda_curva_0001/checkpoint-26000/',
                'output_iccv/coco17_proposed_lambda_curva_001/checkpoint-26000/',
                'output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/',
                'output_iccv/coco17_proposed_lambda_curva_1/checkpoint-26000/']
    
    # Extract metrics
    meteor_scores, rouge1_scores = extract_metrics_from_folder_list(folder_list)
    
    print(f"Found {len(meteor_scores)} METEOR scores and {len(rouge1_scores)} ROUGE-1 scores")
    print(meteor_scores)
    print(rouge1_scores)
    # Save metrics to file
    # output_file = input("Enter the output file path (default: metrics_results.csv): ") or "metrics_results.csv"
    # save_metrics_to_file(folder_list, meteor_scores, rouge1_scores, output_file)
    
    # print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()