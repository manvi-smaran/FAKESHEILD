"""Training module initialization."""
from .lora_finetune import LoRATrainer, run_lora_finetuning
from .peft_finetune import PEFTTrainer, run_peft_finetuning
from .peft_configs import (
    PEFT_METHODS,
    create_peft_config,
    list_available_methods,
    get_method_info,
)

__all__ = [
    # Legacy LoRA exports
    "LoRATrainer",
    "run_lora_finetuning",
    # New unified PEFT exports
    "PEFTTrainer",
    "run_peft_finetuning",
    "PEFT_METHODS",
    "create_peft_config",
    "list_available_methods",
    "get_method_info",
]
