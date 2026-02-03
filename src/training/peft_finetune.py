"""
Unified PEFT Fine-Tuning Module for VLM Deepfake Detection

Supports multiple PEFT methods:
- LoRA: Low-Rank Adaptation (baseline)
- QLoRA: Quantized LoRA (4-bit)
- DoRA: Weight-Decomposed LoRA
- AdaLoRA: Adaptive Rank LoRA
- Prefix Tuning: Learnable prefix tokens
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import yaml
import random

from .peft_configs import (
    PEFT_METHODS,
    create_peft_config,
    get_quantization_config,
    get_method_info,
    list_available_methods,
)


# =============================================================================
# Dataset (reused from lora_finetune)
# =============================================================================

class DeepfakeFineTuneDataset(Dataset):
    """Dataset for PEFT fine-tuning on deepfake detection."""
    
    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        processor,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.split = split
        
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Use existing dataset loader
        from src.data.dataset_loader import create_dataset
        full_dataset = create_dataset(config, dataset_name, max_samples)
        
        # Get all samples
        all_samples = full_dataset.samples
        
        # BALANCE THE DATASET: separate by class
        real_samples = [s for s in all_samples if s[1] == 0]  # label=0 is Real
        fake_samples = [s for s in all_samples if s[1] == 1]  # label=1 is Fake
        
        # Shuffle each class
        random.seed(42)
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        # Take equal number from each class (balanced sampling)
        min_class_size = min(len(real_samples), len(fake_samples))
        balanced_samples = real_samples[:min_class_size] + fake_samples[:min_class_size]
        
        # Shuffle the combined balanced dataset
        random.shuffle(balanced_samples)
        
        print(f"[Dataset] Balanced: {min_class_size} Real + {min_class_size} Fake = {len(balanced_samples)} total")
        
        # Split train/val (80/20)
        split_idx = int(len(balanced_samples) * 0.8)
        if self.split == "train":
            self.samples = balanced_samples[:split_idx]
        else:
            self.samples = balanced_samples[split_idx:]
        
        # Prompt template for training
        self.prompt = """Analyze this image of a human face carefully.
Is this a real photograph or a deepfake/AI-manipulated image?
Answer with 'Real' or 'Fake' followed by a brief explanation."""
        
        # Response templates
        self.responses = {
            0: "Real. This appears to be an authentic photograph with natural skin texture, consistent lighting, and no visible manipulation artifacts.",
            1: "Fake. This image shows signs of deepfake manipulation including subtle blending artifacts around facial boundaries and inconsistencies in texture."
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label, manip_type = self.samples[idx]
        
        # Load image
        image = Image.open(frame_path).convert("RGB")
        
        # Create conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": self.responses[label]}
                ]
            }
        ]
        
        return {
            "image": image,
            "messages": messages,
            "label": label,
        }


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "smolvlm": {
        "model_name": "HuggingFaceTB/SmolVLM-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "processor_class": "AutoProcessor",
        "model_class": "AutoModelForVision2Seq",
    },
    "qwen_vl": {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "processor_class": "AutoProcessor", 
        "model_class": "Qwen2VLForConditionalGeneration",
    },
    # NOTE: Moondream uses a custom API (encode_image/answer_question) that's
    # incompatible with standard PEFT training. It can still be used for inference.
}


# =============================================================================
# Unified PEFT Trainer
# =============================================================================

class PEFTTrainer:
    """Unified PEFT trainer supporting multiple methods and VLM architectures."""
    
    def __init__(
        self,
        model_type: str = "smolvlm",
        peft_method: str = "lora",
        output_dir: str = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        **peft_kwargs,
    ):
        # Validate model type
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
        
        # Validate PEFT method
        if peft_method not in PEFT_METHODS:
            raise ValueError(f"Unknown PEFT method: {peft_method}. Choose from {list_available_methods()}")
        
        self.model_type = model_type
        self.peft_method = peft_method
        self.model_config = MODEL_CONFIGS[model_type]
        self.peft_info = PEFT_METHODS[peft_method]
        
        # Set output directory
        self.output_dir = Path(output_dir or f"checkpoints/{model_type}-{peft_method}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.peft_kwargs = peft_kwargs
        
        # Model components (initialized in setup)
        self.model = None
        self.processor = None
        self.config_path = "configs/model_configs.yaml"
        self.dataset_name = "celebdf"
        
        # Datasets (cached after first creation)
        self._train_dataset = None
        self._val_dataset = None
        
        print(f"Initialized PEFTTrainer:")
        print(f"  Model: {self.model_type} ({self.model_config['model_name']})")
        print(f"  PEFT Method: {get_method_info(peft_method)}")
        print(f"  Output: {self.output_dir}")
    
    def setup(self):
        """Load model and apply PEFT configuration."""
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
        from peft import get_peft_model
        
        print(f"\nLoading {self.model_type} processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_config["model_name"],
            trust_remote_code=True,
        )
        
        print(f"Loading {self.model_type} model...")
        
        # Check if quantization needed (QLoRA)
        quantization_config = None
        if self.peft_info.requires_quantization:
            print("Enabling 4-bit quantization for QLoRA...")
            quantization_config = get_quantization_config()
        
        # Model-specific loading
        if self.model_type == "qwen_vl":
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_config["model_name"],
                torch_dtype=torch.bfloat16 if quantization_config else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        elif self.model_type == "moondream":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["model_name"],
                torch_dtype=torch.bfloat16 if quantization_config else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_config["model_name"],
                torch_dtype=torch.bfloat16 if quantization_config else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        
        # Create PEFT config
        print(f"\nApplying {self.peft_info.name} configuration...")
        peft_config = create_peft_config(
            peft_method=self.peft_method,
            target_modules=self.model_config["target_modules"],
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            **self.peft_kwargs,
        )
        
        # Apply PEFT
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Enable training mode
        self.model.train()
        
        # Ensure PEFT parameters require gradient
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if any(key in name.lower() for key in ['lora', 'prefix', 'adapter']):
                param.requires_grad = True
                trainable_count += 1
        
        print(f"{self.peft_info.name} setup complete! ({trainable_count} trainable parameter groups)")
    
    def create_dataloaders(self, max_samples: Optional[int] = None, force_reload: bool = False):
        """Create train and validation dataloaders (cached to ensure consistency)."""
        # Use cached datasets if available (ensures train/val consistency)
        if self._train_dataset is not None and self._val_dataset is not None and not force_reload:
            print(f"Using cached datasets - Train: {len(self._train_dataset)}, Val: {len(self._val_dataset)}")
            return self._train_dataset, self._val_dataset
        
        self._train_dataset = DeepfakeFineTuneDataset(
            config_path=self.config_path,
            dataset_name=self.dataset_name,
            processor=self.processor,
            split="train",
            max_samples=max_samples,
        )
        
        self._val_dataset = DeepfakeFineTuneDataset(
            config_path=self.config_path,
            dataset_name=self.dataset_name,
            processor=self.processor,
            split="val",
            max_samples=max_samples,
        )
        
        print(f"Train samples: {len(self._train_dataset)}")
        print(f"Val samples: {len(self._val_dataset)}")
        
        return self._train_dataset, self._val_dataset
    
    def train(self, max_samples: Optional[int] = None):
        """Run PEFT fine-tuning."""
        if self.model is None:
            self.setup()
        
        train_dataset, val_dataset = self.create_dataloaders(max_samples)
        
        # Collate function for batching - model-specific processing
        def collate_fn(examples):
            images = [ex["image"] for ex in examples]
            messages_list = [ex["messages"] for ex in examples]
            labels_list = [ex["label"] for ex in examples]
            
            # Process each example
            all_input_ids = []
            all_attention_mask = []
            all_pixel_values = []
            all_labels = []
            all_image_grid_thw = []  # For Qwen2-VL
            
            for image, messages in zip(images, messages_list):
                # SmolVLM, Qwen2-VL use chat template
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                
                # Process inputs
                inputs = self.processor(
                    text=text,
                    images=[image],
                    return_tensors="pt",
                )
                
                all_input_ids.append(inputs["input_ids"].squeeze(0))
                all_attention_mask.append(inputs["attention_mask"].squeeze(0))
                
                if "pixel_values" in inputs:
                    all_pixel_values.append(inputs["pixel_values"].squeeze(0))
                
                # Qwen2-VL specific: image_grid_thw
                if "image_grid_thw" in inputs:
                    all_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))
                
                # Create labels (same as input_ids for causal LM)
                label_ids = inputs["input_ids"].squeeze(0).clone()
                all_labels.append(label_ids)
            
            # Pad sequences
            max_len = max(ids.size(0) for ids in all_input_ids)
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            pad_token_id = self.processor.tokenizer.pad_token_id or 0
            
            for input_ids, attention_mask, label_ids in zip(all_input_ids, all_attention_mask, all_labels):
                pad_len = max_len - input_ids.size(0)
                padded_input_ids.append(
                    torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                )
                padded_attention_mask.append(
                    torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                )
                padded_labels.append(
                    torch.cat([label_ids, torch.full((pad_len,), -100, dtype=label_ids.dtype)])
                )
            
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
            }
            
            if all_pixel_values:
                batch["pixel_values"] = torch.stack(all_pixel_values)
            
            # Qwen2-VL specific: add image_grid_thw
            if all_image_grid_thw:
                batch["image_grid_thw"] = torch.stack(all_image_grid_thw)
            
            return batch
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # Training loop
        print(f"\nStarting {self.peft_info.name} Fine-Tuning")
        print(f"Epochs: {self.num_epochs}, Batch Size: {self.batch_size}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print("=" * 50)
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Gradient accumulation step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                progress_bar.set_postfix({"loss": f"{epoch_loss / num_batches:.4f}"})
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"New best loss! Saving checkpoint...")
                self.model.save_pretrained(self.output_dir / "best")
        
        # Save final checkpoint
        print(f"\nSaving final {self.peft_info.name} adapter...")
        self.model.save_pretrained(self.output_dir / "final")
        
        # Save training info
        training_info = {
            "model_type": self.model_type,
            "peft_method": self.peft_method,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "best_loss": best_loss,
            "final_loss": avg_loss,
        }
        
        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training complete! Checkpoints saved to {self.output_dir}")
        
        return training_info
    
    def evaluate(self, max_samples: int = None):
        """Evaluate the fine-tuned model using the same val dataset from training."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        self.model.eval()
        
        # Reuse cached datasets (don't reload with different max_samples)
        _, val_dataset = self.create_dataloaders()
        
        correct = 0
        total = 0
        
        # Limit evaluation samples if specified
        eval_samples = len(val_dataset) if max_samples is None else min(len(val_dataset), max_samples)
        
        print(f"\nEvaluating {self.peft_info.name} model on {eval_samples} samples...")
        
        with torch.no_grad():
            for i in tqdm(range(eval_samples)):
                sample = val_dataset[i]
                image = sample["image"]
                label = sample["label"]
                
                # Model-specific text preparation for inference
                # SmolVLM, Qwen2-VL use chat template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": val_dataset.prompt}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                
                inputs = self.processor(
                    text=text,
                    images=[image],
                    return_tensors="pt",
                )
                
                # Move to device
                inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
                
                # Parse prediction - only look at the NEW generated text
                # Get the input length to extract only generated tokens
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_len:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)
                
                # Debug: show first 3 predictions
                if i < 3:
                    print(f"\n[DEBUG] Sample {i}: Label={label}, Generated='{response[:100]}...'")
                
                # Parse prediction
                response_lower = response.lower()
                if "fake" in response_lower:
                    pred = 1
                elif "real" in response_lower:
                    pred = 0
                else:
                    pred = -1
                    if i < 3:
                        print(f"[DEBUG] Sample {i}: UNPARSEABLE - pred=-1")
                
                if pred == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        # Debug: Print label distribution
        fake_count = sum(1 for i in range(min(eval_samples, len(val_dataset))) if val_dataset[i]["label"] == 1)
        real_count = eval_samples - fake_count
        print(f"  Label distribution - Real: {real_count}, Fake: {fake_count}")
        
        return {"accuracy": accuracy, "correct": correct, "total": total}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_peft_finetuning(
    config_path: str = "configs/model_configs.yaml",
    dataset_name: str = "celebdf",
    output_dir: str = None,
    max_samples: Optional[int] = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    model_type: str = "smolvlm",
    peft_method: str = "lora",
    **peft_kwargs,
):
    """
    Main function to run PEFT fine-tuning.
    
    Args:
        model_type: One of 'smolvlm', 'qwen_vl', 'moondream'
        peft_method: One of 'lora', 'qlora', 'dora', 'adalora', 'prefix'
    """
    output_dir = output_dir or f"checkpoints/{model_type}-{peft_method}"
    
    trainer = PEFTTrainer(
        model_type=model_type,
        peft_method=peft_method,
        output_dir=output_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        **peft_kwargs,
    )
    
    trainer.config_path = config_path
    trainer.dataset_name = dataset_name
    
    training_info = trainer.train(max_samples=max_samples)
    
    metrics = trainer.evaluate()
    
    return trainer, {**training_info, **metrics}
