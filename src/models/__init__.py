from .base_vslm import BaseVSLM
from .qwen_vl import QwenVL
from .smolvlm import SmolVLM
from .moondream import MoonDream
from .florence2 import Florence2
from .llava_next import LLaVANext

MODEL_REGISTRY = {
    "qwen_vl": QwenVL,
    "smolvlm": SmolVLM,
    "moondream": MoonDream,
    "florence2": Florence2,
    "llava_next": LLaVANext,
}

def get_model(model_name: str, config: dict) -> BaseVSLM:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
