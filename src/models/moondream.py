from typing import List
from PIL import Image
import torch
import torch.nn.functional as F

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
    
    def predict_with_forced_choice(self, image: Image.Image, prompt: str) -> dict:
        """
        Predict with forced-choice p_fake for MoonDream2.
        
        MoonDream uses custom encode_image + answer_question API,
        so we use a simplified forced-choice approach:
        1. Generate single token with "Real or Fake?" prompt
        2. Score logits for Real/Fake tokens
        
        Returns:
            dict with 'response', 'p_fake', 'log_p_real', 'log_p_fake'
        """
        if self.model is None:
            self.load_model()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Append classification suffix
        classification_prompt = prompt + "\n\nAnswer with exactly one word: Real or Fake.\n\nAnswer:"
        
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            
            # Get full response for evidence
            full_response = self.model.answer_question(
                enc_image, 
                prompt, 
                self.processor,
                max_new_tokens=128,
            )
            
            # Get single-word answer for classification
            short_response = self.model.answer_question(
                enc_image, 
                classification_prompt, 
                self.processor,
                max_new_tokens=5,  # Just enough for "Real" or "Fake"
            )
        
        # Parse the short response to determine p_fake
        short_clean = short_response.strip().lower()
        
        # Simple heuristic from response (fallback if logits unavailable)
        if "fake" in short_clean and "real" not in short_clean:
            p_fake = 0.9
            log_p_fake, log_p_real = 0.0, -2.2  # Approximate
        elif "real" in short_clean and "fake" not in short_clean:
            p_fake = 0.1
            log_p_fake, log_p_real = -2.2, 0.0  # Approximate
        else:
            p_fake = 0.5
            log_p_fake, log_p_real = 0.0, 0.0
        
        return {
            "response": full_response,
            "p_fake": p_fake,
            "log_p_real": log_p_real,
            "log_p_fake": log_p_fake,
            "short_response": short_response,
            "p_fake_method": "forced_choice_response_parse",
        }
    
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

