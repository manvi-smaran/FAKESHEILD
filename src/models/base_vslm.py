from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from PIL import Image
import torch


class BaseVSLM(ABC):
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        pass
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        return [self.predict(img, prompt) for img in images]
    
    def get_quantization_config(self):
        from transformers import BitsAndBytesConfig
        
        quant_type = self.config.get("quantization", "none")
        
        if quant_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quant_type == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def get_torch_dtype(self):
        dtype_str = self.config.get("torch_dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        response_lower = response.lower()
        
        if "fake" in response_lower or "manipulated" in response_lower or "deepfake" in response_lower:
            prediction = "fake"
            confidence = 0.8
        elif "real" in response_lower or "authentic" in response_lower or "genuine" in response_lower:
            prediction = "real"
            confidence = 0.8
        else:
            prediction = "uncertain"
            confidence = 0.5
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_response": response
        }
    
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
