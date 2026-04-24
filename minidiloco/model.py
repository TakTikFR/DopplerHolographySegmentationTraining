import os
from transformers import AutoModelForCausalLM, AutoConfig


def get_gpt2_model():
    """Return a fresh GPT-2 model placed on the correct CUDA device."""
    config = AutoConfig.from_pretrained("gpt2")
    rank = int(os.environ["LOCAL_RANK"])
    return AutoModelForCausalLM.from_config(config).cuda(rank)
