from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class SmolVLM(BaseVSLM):
    """SmolVLM-Instruct wrapper - a small and efficient vision-language model."""
    
    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        self.processor = AutoProcessor.from_pretrained(
            self.config["hub_id"],
            trust_remote_code=True,
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
    
    def predict(self, image: Image.Image, prompt: str) -> str:
        if self.model is None:
            self.load_model()
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
            )
        
        # Trim input tokens
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        return output_text.strip()
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        if self.model is None:
            self.load_model()
        
        results = []
        for img in images:
            results.append(self.predict(img, prompt))
        
        return results
