from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class QwenVL(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        quant_config = self.get_quantization_config()
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config["hub_id"],
                torch_dtype=self.get_torch_dtype(),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception as e:
            print(f"Quantized loading failed: {e}")
            print("Falling back to FP16 without quantization...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text
    
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
