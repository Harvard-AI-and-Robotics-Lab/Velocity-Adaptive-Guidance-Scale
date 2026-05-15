#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 96:00:00
##SBATCH -p nvidia
##SBATCH --gres=gpu:a100:2
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --mail-type=END
#SBATCH --job-name='train'
##SBATCH --mail-user=user@example.com
#SBATCH --output=train_%j.out

# source ~/.bashrc
# conda activate torch2.4_cuda11.8

source activate base
conda activate flowmatching

# HUGGINGFACE_TOKEN=''
HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
# IMAGE_WIDTH=768
# IMAGE_HEIGHT=768
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

PROMPT_DIR="/path/to/datasets/richhf_18k_dataset/richhf_18k/test_for_simpletuner"
PROMPT_DIR="/path/to/datasets/coco17/validation"
# PROMPT_DIR="/path/to/data/richhf_18k/test_for_simpletuner"
OUTPUT_DIR='generated_images_coco17'

# MODEL_PATH='output/sd_3_5_finetune_lr_1_5/checkpoint-30000'
# MODEL_PATH=(
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_0001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_01/checkpoint-1000'
#             )
MODEL_PATH=(
            # 'output_iccv/coco17_no_reweight/checkpoint-26000'
            # 'output_iccv/coco17_lognorm/checkpoint-26000'
            # 'output_iccv/coco17_modesample/checkpoint-26000'
            'output_iccv/coco17_cosmap/checkpoint-26000'
            # 'output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000'
            # 'output_iccv/coco17_proposed_lambda_curva_0/checkpoint-26000'
            # 'output_iccv/coco17_proposed_lambda_curva_001/checkpoint-26000'
            # 'output_iccv/coco17_proposed_lambda_curva_0001/checkpoint-26000'
            # 'output_iccv/coco17_proposed_lambda_curva_1/checkpoint-26000'
            )

OUTPUT_DIR=(
    # "data4paper/intermediate_noise/coco17_no_reweight"
    # "data4paper/intermediate_noise/coco17_lognorm"
    # "data4paper/intermediate_noise/coco17_modesample"
    "data4paper/intermediate_noise/coco17_cosmap"
    # "data4paper/intermediate_noise/coco17_proposed_lambda_curva_01"
)

for (( i=0; i<${#MODEL_PATH[@]}; i++ ));
do
    python analysis/visualize_intermediate_noise.py \
    --caption_folder=${PROMPT_DIR} \
    --model_path=${MODEL_PATH[$i]} \
    --output_folder=${OUTPUT_DIR[$i]} 
    # python analysis/visualize_intermediate_noise.py --caption_folder /path/to/captions --output_folder /path/to/output
done