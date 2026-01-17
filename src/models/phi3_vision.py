from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class Phi3Vision(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        quant_config = self.get_quantization_config()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["hub_id"],
                torch_dtype=self.get_torch_dtype(),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
                _attn_implementation="eager",
            )
        except Exception as e:
            print(f"Quantized loading failed: {e}")
            print("Falling back to FP16...")
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
        
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{prompt}",
            }
        ]
        
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
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
