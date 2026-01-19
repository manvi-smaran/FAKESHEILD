from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class Florence2(BaseVSLM):
    """Florence-2 wrapper - Microsoft's efficient vision-language model."""
    
    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        self.processor = AutoProcessor.from_pretrained(
            self.config["hub_id"],
            trust_remote_code=True,
        )
        
        # Load with explicit attention implementation for compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid SDPA issues
        )
        self.model.eval()
    
    def predict(self, image: Image.Image, prompt: str) -> str:
        if self.model is None:
            self.load_model()
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Florence-2 uses specific task prompts
        # For general VQA, we wrap the prompt
        task_prompt = f"<VQA>{prompt}"
        
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
                num_beams=3,
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        # Remove prompt from response if present
        if generated_text.startswith(task_prompt):
            generated_text = generated_text[len(task_prompt):].strip()
        
        return generated_text
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        if self.model is None:
            self.load_model()
        
        results = []
        for img in images:
            results.append(self.predict(img, prompt))
        
        return results
