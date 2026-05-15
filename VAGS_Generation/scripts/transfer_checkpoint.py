#!/usr/bin/env python3
import os
import torch
import argparse
import json
from safetensors.torch import load_file, save_file
from accelerate import Accelerator
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from helpers.training.trainer import Trainer as TrainerOld
from helpers.training.trainer_proposed import Trainer as TrainerProposed
from helpers.configuration.loader import load_config

# apply parse_args  to the script
parser = argparse.ArgumentParser(description="Transfer checkpoint")
args = parser.parse_args()

trainer = TrainerOld()

# accelerator = Accelerator(
#                 gradient_accumulation_steps=1,
#                 mixed_precision=(
#                     "bf16"
#                     if not torch.backends.mps.is_available()
#                     else None
#                 )
#             )
# accelerator.load_state(os.path.join(self.config.output_dir, path))