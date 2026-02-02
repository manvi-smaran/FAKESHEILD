"""
PEFT Configuration Factory for Multiple Fine-Tuning Methods

Supports: LoRA, QLoRA, DoRA, AdaLoRA, Prefix Tuning
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class PEFTMethodConfig:
    """Configuration for a PEFT method."""
    name: str
    description: str
    requires_quantization: bool = False
    default_params: Dict[str, Any] = None


# =============================================================================
# PEFT Method Definitions
# =============================================================================

PEFT_METHODS = {
    "lora": PEFTMethodConfig(
        name="LoRA",
        description="Low-Rank Adaptation - baseline adapter method",
        requires_quantization=False,
        default_params={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
        }
    ),
    "qlora": PEFTMethodConfig(
        name="QLoRA",
        description="Quantized LoRA - 4-bit quantization for memory efficiency",
        requires_quantization=True,
        default_params={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
        }
    ),
    "dora": PEFTMethodConfig(
        name="DoRA",
        description="Weight-Decomposed LoRA - improved LoRA with magnitude/direction decomposition",
        requires_quantization=False,
        default_params={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "use_dora": True,
        }
    ),
    "adalora": PEFTMethodConfig(
        name="AdaLoRA",
        description="Adaptive LoRA - dynamically allocates rank budget during training",
        requires_quantization=False,
        default_params={
            "init_r": 12,
            "target_r": 8,
            "deltaT": 10,
            "beta1": 0.85,
            "beta2": 0.85,
            "orth_reg_weight": 0.5,
        }
    ),
    "prefix": PEFTMethodConfig(
        name="Prefix Tuning",
        description="Prepends learnable prefix tokens to each layer",
        requires_quantization=False,
        default_params={
            "num_virtual_tokens": 20,
            "prefix_projection": True,
        }
    ),
}


def get_quantization_config():
    """Get BitsAndBytes config for 4-bit quantization (QLoRA)."""
    from transformers import BitsAndBytesConfig
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def create_peft_config(
    peft_method: str,
    target_modules: list,
    rank: int = 16,
    alpha: int = 32,
    **kwargs
):
    """
    Create PEFT configuration based on method type.
    
    Args:
        peft_method: One of 'lora', 'qlora', 'dora', 'adalora', 'prefix'
        target_modules: List of module names to apply PEFT to
        rank: LoRA rank (r parameter)
        alpha: LoRA alpha scaling factor
        **kwargs: Additional method-specific parameters
    
    Returns:
        PEFT config object
    """
    from peft import LoraConfig, TaskType
    
    if peft_method not in PEFT_METHODS:
        raise ValueError(f"Unknown PEFT method: {peft_method}. Choose from {list(PEFT_METHODS.keys())}")
    
    method_info = PEFT_METHODS[peft_method]
    
    if peft_method in ["lora", "qlora"]:
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=kwargs.get("lora_dropout", 0.05),
            bias=kwargs.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
    
    elif peft_method == "dora":
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=kwargs.get("lora_dropout", 0.05),
            bias=kwargs.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,  # Key flag for DoRA
        )
    
    elif peft_method == "adalora":
        from peft import AdaLoraConfig
        
        return AdaLoraConfig(
            init_r=kwargs.get("init_r", 12),
            target_r=kwargs.get("target_r", rank),
            deltaT=kwargs.get("deltaT", 10),
            beta1=kwargs.get("beta1", 0.85),
            beta2=kwargs.get("beta2", 0.85),
            orth_reg_weight=kwargs.get("orth_reg_weight", 0.5),
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=kwargs.get("lora_dropout", 0.05),
            task_type=TaskType.CAUSAL_LM,
        )
    
    elif peft_method == "prefix":
        from peft import PrefixTuningConfig
        
        return PrefixTuningConfig(
            num_virtual_tokens=kwargs.get("num_virtual_tokens", 20),
            prefix_projection=kwargs.get("prefix_projection", True),
            task_type=TaskType.CAUSAL_LM,
        )
    
    else:
        raise ValueError(f"PEFT method '{peft_method}' not implemented yet")


def get_method_info(peft_method: str) -> str:
    """Get human-readable info about a PEFT method."""
    if peft_method not in PEFT_METHODS:
        return f"Unknown method: {peft_method}"
    
    info = PEFT_METHODS[peft_method]
    return f"{info.name}: {info.description}"


def list_available_methods() -> list:
    """List all available PEFT methods."""
    return list(PEFT_METHODS.keys())
