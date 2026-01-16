from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class MiniCPMV(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        
        quant_config = self.get_quantization_config()
        
        self.model = AutoModel.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.processor = AutoTokenizer.from_pretrained(
            self.config["hub_id"],
            trust_remote_code=True,
        )
    
    def predict(self, image: Image.Image, prompt: str) -> str:
        if self.model is None:
            self.load_model()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        messages = [
            {"role": "user", "content": [image, prompt]}
        ]
        
        with torch.no_grad():
            response = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.processor,
                sampling=False,
                max_new_tokens=self.config.get("max_new_tokens", 256),
            )
        
        return response
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        if self.model is None:
            self.load_model()
        
        results = []
        batch_size = self.config.get("batch_size", 4)
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = [self.predict(img, prompt) for img in batch_images]
            results.extend(batch_results)
        
        return results
