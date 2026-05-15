# --------------------------------------------------------------------------
#
# A script to create a visual report by merging generated images with their
# ground truth and all model evaluation results into single PNG files.
#
# --------------------------------------------------------------------------

import json
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

# --- Metric Calculation (Functions from previous script) ---

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def calculate_sentence_metrics(generated_sentence, ground_truth_sentence):
    """Calculates BLEU, METEOR, and ROUGE scores for a single sentence pair."""
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
    """Reads all JSON files and aggregates their evaluations, keyed by sample ID."""
    aggregated_results = {}
    print("🚀 Starting data aggregation...\n")
    for file_path in file_paths:
        try:
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


# --- Reworked Image Generation Function for Side-by-Side Layout ---

def create_comparison_image(rank, sample_id, data, image_path, output_dir, font_path=None):
    """
    Creates a single image merging a source image and a text block side-by-side.
    """
    # --- 1. Setup Canvas and Fonts ---
    PADDING = 25
    TEXT_AREA_WIDTH = 500 # The width of the text column on the right
    try:
        font_bold = ImageFont.truetype(font_path or "arialbd.ttf", 18)
        font_regular = ImageFont.truetype(font_path or "arial.ttf", 16)
        font_scores = ImageFont.truetype(font_path or "cour.ttf", 15) # Monospaced for scores
    except IOError:
        font_bold = ImageFont.load_default()
        font_regular = ImageFont.load_default()
        font_scores = ImageFont.load_default()

    # --- 2. Load Source Image ---
    try:
        source_img = Image.open(image_path)
    except FileNotFoundError:
        print(f"  -> ⚠️  Warning: Image not found for sample {sample_id} at {image_path}")
        return

    # --- 3. Prepare and Wrap Text ---
    text_blocks = []
    # Ground Truth Text
    text_blocks.append(("Ground Truth:", font_bold))
    gt_wrapped = textwrap.wrap(f"'{data['ground_truth']}'", width=60)
    for line in gt_wrapped:
        text_blocks.append((line, font_regular))
    text_blocks.append(("", font_regular))

    # Model Evaluations Text
    sorted_evals = sorted(data['evaluations'], key=lambda x: x['source_model'])
    for eval_item in sorted_evals:
        text_blocks.append((f"▶ Model: {eval_item['source_model']}", font_bold))
        gen_wrapped = textwrap.wrap(f"'{eval_item['generated']}'", width=60)
        for line in gen_wrapped:
            text_blocks.append((line, font_regular))

        metrics = eval_item['metrics']
        scores_text = (f"  Scores  |  BLEU-4: {metrics['BLEU-4']:5.2f}  |  "
                       f"METEOR: {metrics['METEOR']:5.2f}  |  "
                       f"ROUGE-L: {metrics['ROUGE-L']:5.2f}")
        text_blocks.append((scores_text, font_scores))
        text_blocks.append(("", font_regular))

    # --- 4. Calculate Final Dimensions ---
    text_height = sum([font.getbbox(text)[3] + 6 for text, font in text_blocks])
    canvas_width = PADDING + source_img.width + PADDING + TEXT_AREA_WIDTH + PADDING
    canvas_height = PADDING + max(source_img.height, text_height) + PADDING
    
    # --- 5. Create Canvas and Draw ---
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # Paste the image on the left
    canvas.paste(source_img, (PADDING, PADDING))

    # Draw the text block on the right
    x_cursor = PADDING + source_img.width + PADDING
    y_cursor = PADDING
    for text, font in text_blocks:
        draw.text((x_cursor, y_cursor), text, fill="black", font=font)
        y_cursor += font.getbbox(text)[3] + 6

    # --- 6. Save Result ---
    output_filename = f"{rank+1:04d}_{sample_id}.png"
    canvas.save(os.path.join(output_dir, output_filename))


if __name__ == "__main__":
    # Define the list of your input JSON files
    input_files = [
        'output_baseline/coco17_sd35_no_reweight/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_lognorm/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_modesample/checkpoint-26000/coco17_caption_results.json',
        'output_baseline/coco17_sd35_cosmap/checkpoint-26000/coco17_caption_results.json',
        'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/coco17_caption_results.json'
    ]

    output_dir = "data4paper/visual_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Aggregate and Sort Data ---
    aggregated_data = aggregate_evaluations_by_id(input_files)

    if aggregated_data:
        # Identify the model to sort by (the last one in the list)
        last_file_path = input_files[-1]
        sort_model_name = os.path.basename(os.path.dirname(os.path.dirname(last_file_path)))
        print(f"\n🔃 Sorting results by descending METEOR score from model: '{sort_model_name}'...")

        def get_sort_key(item):
            _sample_id, data = item
            for eval_item in data['evaluations']:
                if eval_item['source_model'] == sort_model_name:
                    return eval_item['metrics']['METEOR']
            return -1

        sorted_items = sorted(aggregated_data.items(), key=get_sort_key, reverse=True)
        
        # --- 2. Generate an Image for Each Sample ---
        print(f"\n🖼️ Generating visual report in '{output_dir}' folder...")
        # Path to the folder containing the *.png files to be merged
        image_base_dir = os.path.join(os.path.dirname(last_file_path), 'generated_images_coco17')

        for rank, (sample_id, data) in enumerate(sorted_items):
            print(f"  - Creating image for sample: {sample_id} (Rank #{rank+1})")
            image_path = os.path.join(image_base_dir, f"{sample_id}.png")
            create_comparison_image(rank, sample_id, data, image_path, output_dir)

        print(f"\n🎉 Success! Visual report has been saved to '{output_dir}'.")