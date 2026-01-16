from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class MoonDream(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
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
        
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            response = self.model.answer_question(
                enc_image, 
                prompt, 
                self.processor,
                max_new_tokens=self.config.get("max_new_tokens", 256),
            )
        
        return response
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        if self.model is None:
            self.load_model()
        
        results = []
        batch_size = self.config.get("batch_size", 8)
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = [self.predict(img, prompt) for img in batch_images]
            results.extend(batch_results)
        
        return results
