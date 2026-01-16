from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class LLaVANext(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        quant_config = self.get_quantization_config()
        
        self.processor = LlavaNextProcessor.from_pretrained(
            self.config["hub_id"],
        )
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
    
    def predict(self, image: Image.Image, prompt: str) -> str:
        if self.model is None:
            self.load_model()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
            )
        
        response = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
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
