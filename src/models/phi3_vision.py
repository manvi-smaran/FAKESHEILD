from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class Phi3Vision(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import AutoProcessor
        
        quant_config = self.get_quantization_config()
        
        # Try different model loading approaches
        try:
            # First try: AutoModelForCausalLM with generate support
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["hub_id"],
                torch_dtype=self.get_torch_dtype(),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception as e:
            print(f"First loading attempt failed: {e}")
            # Fallback: FP16 without quantization
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["hub_id"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        
        self.processor = AutoProcessor.from_pretrained(
            self.config["hub_id"],
            trust_remote_code=True,
        )
    
    def predict(self, image: Image.Image, prompt: str) -> str:
        if self.model is None:
            self.load_model()
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create messages with image placeholder
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        
        # Apply chat template
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
            )
        
        # Trim input tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        
        output_text = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text
    
    def batch_predict(self, images: List[Image.Image], prompt: str) -> List[str]:
        if self.model is None:
            self.load_model()
        
        results = []
        for img in images:
            results.append(self.predict(img, prompt))
        
        return results
