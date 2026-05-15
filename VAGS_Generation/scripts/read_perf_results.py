# python scripts/read_perf_results.py /path/to/folder
# python scripts/read_perf_results.py 'output_iccv/coco17_no_reweight/checkpoint-26000'
# python scripts/read_perf_results.py 'output_iccv/coco17_lognorm/checkpoint-26000'
# python scripts/read_perf_results.py 'output_iccv/coco17_modesample/checkpoint-26000'
# python scripts/read_perf_results.py 'output_iccv/coco17_cosmap/checkpoint-26000'
# python scripts/read_perf_results.py 'output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000'
import json
import os
import re

def extract_all_metrics(folder_path, results_list=['coco17_results.txt', 'coco17_imgreward_results.json', 'coco17_pickscore_results.txt']):
    """
    Extract metrics from three different files within a specified folder:
    1. coco17_results.txt - Extract values like "21.70 & 34.05 & 32.24" with 2nd and 3rd values divided by 100
    2. coco17_imgreward_results.json - Extract "average" score from the statistics
    3. coco17_pickscore_results.txt - Extract "Average PickScore" value
    
    Args:
        folder_path (str): Path to the folder containing the files
        
    Returns:
        list: All extracted metrics in order
    """
    metrics = []
    
    # 1. Parse coco17_results.txt
    results_file = os.path.join(folder_path, results_list[0])
    try:
        with open(results_file, 'r') as f:
            for line in f:
                # Look for lines with the pattern like "21.70 & 34.05 & 32.24"
                if '&' in line:
                    # Split by '&' and strip whitespace
                    values = [val.strip() for val in line.split('&')]
                    if len(values) >= 3:
                        try:
                            # Convert to float, divide 2nd and 3rd values by 100
                            metrics.append(round(float(values[0]), 2))
                            metrics.append(round(float(values[1]) / 100, 4))
                            metrics.append(round(float(values[2]) / 100, 4))
                            break  # Assuming we only need the first matching line
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Warning: {results_file} not found")
    
    # 2. Parse coco17_imgreward_results.json
    imgreward_file = os.path.join(folder_path, results_list[1])
    try:
        with open(imgreward_file, 'r') as f:
            data = json.load(f)
            # Extract average score from statistics
            if 'statistics' in data and 'average' in data['statistics']:
                metrics.append(round(float(data['statistics']['average']), 4))
    except FileNotFoundError:
        print(f"Warning: {imgreward_file} not found")
    except json.JSONDecodeError:
        print(f"Warning: Error parsing JSON in {imgreward_file}")
    
    # 3. Parse coco17_pickscore_results.txt
    pickscore_file = os.path.join(folder_path, results_list[2])
    try:
        with open(pickscore_file, 'r') as f:
            for line in f:
                # Look for "Average PickScore:" pattern
                match = re.search(r'Average PickScore:\s*(\d+\.\d+)', line)
                if match:
                    metrics.append(round(float(match.group(1)), 2))
                    break
    except FileNotFoundError:
        print(f"Warning: {pickscore_file} not found")
    
    return metrics

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter the folder path containing the COCO17 result files: ")
    
    # results_list = ['coco17_results.txt', 'coco17_imgreward_results.json', 'coco17_pickscore_results.txt']
    results_list = ['coco14_results.txt', 'coco14_imgreward_results.json', 'coco14_pickscore_results.txt']

    results = extract_all_metrics(folder_path, results_list)
    output_str = ' & '.join([str(val) for val in results])
    print(output_str)