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
    
    def predict_with_forced_choice(
        self, 
        image: Image.Image, 
        prompt: str,
        evidence_prompt: str = None,
    ) -> dict:
        """
        Predict with forced-choice p_fake using full sequence log-probability.
        
        This is the reviewer-proof approach:
        - Scores P("Real" | prompt, image) vs P("Fake" | prompt, image)
        - Uses sigmoid(log_p_fake - log_p_real) for stability
        - Separate prompts for scoring vs evidence generation
        
        Args:
            image: Input image
            prompt: Base detection prompt
            evidence_prompt: Optional separate prompt for evidence (defaults to prompt)
        
        Returns:
            dict with 'response', 'p_fake', 'pred', 'log_p_real', 'log_p_fake'
        """
        from src.utils.forced_choice import (
            score_best_candidate, 
            append_classification_suffix,
            REAL_CANDIDATES,
            FAKE_CANDIDATES,
        )
        
        if self.model is None:
            self.load_model()
        
        # Use separate prompts for scoring vs evidence
        scoring_prompt = append_classification_suffix(prompt)
        ev_prompt = evidence_prompt if evidence_prompt else prompt
        
        # Build scoring inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": scoring_prompt},
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
        
        # Ensure model is in eval mode for reproducibility
        self.model.eval()
        
        # Score with whitespace variants for tokenizer robustness
        try:
            log_p_real, best_real = score_best_candidate(
                self.model, inputs, self.processor.tokenizer, REAL_CANDIDATES
            )
            log_p_fake, best_fake = score_best_candidate(
                self.model, inputs, self.processor.tokenizer, FAKE_CANDIDATES
            )
            
            # Stable: sigmoid(log_p_fake - log_p_real)
            p_fake = torch.sigmoid(torch.tensor(log_p_fake - log_p_real)).item()
            
        except Exception as e:
            print(f"Forced-choice scoring failed: {e}")
            log_p_real, log_p_fake, p_fake = 0.0, 0.0, 0.5
            best_real, best_fake = "Real", "Fake"
        
        # Derive label from p_fake for consistency
        pred = int(p_fake >= 0.5)
        
        # Generate evidence with separate prompt (better quality)
        evidence_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ev_prompt},
                ],
            }
        ]
        
        ev_text = self.processor.apply_chat_template(
            evidence_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        ev_inputs = self.processor(
            text=[ev_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            gen_ids = self.model.generate(
                **ev_inputs, 
                max_new_tokens=128, 
                do_sample=False
            )
            response = self.processor.batch_decode(
                gen_ids[:, ev_inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )[0]
        
        return {
            "response": response,
            "p_fake": p_fake,
            "pred": pred,
            "log_p_real": log_p_real,
            "log_p_fake": log_p_fake,
            "p_fake_method": "forced_choice_full_sequence",
        }
    
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


