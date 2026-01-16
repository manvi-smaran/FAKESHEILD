from typing import List
from PIL import Image
import torch

from .base_vslm import BaseVSLM


class InternVL(BaseVSLM):
    
    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        
        quant_config = self.get_quantization_config()
        
        self.model = AutoModel.from_pretrained(
            self.config["hub_id"],
            torch_dtype=self.get_torch_dtype(),
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
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
        
        pixel_values = self._load_image(image)
        
        generation_config = dict(
            max_new_tokens=self.config.get("max_new_tokens", 256),
            do_sample=False,
        )
        
        with torch.no_grad():
            response = self.model.chat(
                self.processor,
                pixel_values,
                prompt,
                generation_config,
            )
        
        return response
    
    def _load_image(self, image: Image.Image):
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values.to(self.device, dtype=self.get_torch_dtype())
    
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
