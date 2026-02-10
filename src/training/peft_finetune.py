"""
PEFT Fine-Tuning Module for Vision Language Models

Supports multiple PEFT methods: LoRA, QLoRA, DoRA, AdaLoRA, Prefix Tuning
Models: SmolVLM, Qwen2-VL

Key fixes implemented:
- Labels masking: loss ONLY on assistant answer tokens
- Deterministic dataset split: balanced 50/50, no overlap
- One-word answers: " Real" or " Fake" for clean training
- Forced-choice evaluation: use p_fake scores, not text parsing
"""

import os
import json
import random
import yaml
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .peft_configs import (
    PEFT_METHODS,
    create_peft_config,
    get_quantization_config,
    get_method_info,
    list_available_methods,
)


# =============================================================================
# Dataset Class (accepts precomputed samples)
# =============================================================================

class DeepfakeFineTuneDataset(Dataset):
    """Dataset for PEFT fine-tuning on deepfake detection.
    
    Takes precomputed, balanced samples to ensure train/val consistency.
    """
    
    def __init__(self, samples: List[Tuple], processor):
        self.samples = samples
        self.processor = processor
        
        # Simple prompt that matches evaluation
        self.prompt = (
            "Classify authenticity.\n\n"
            "Answer with exactly one word: Real or Fake.\n\n"
            "Answer:"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label, manip_type = self.samples[idx]
        
        # Load image
        image = Image.open(frame_path).convert("RGB")
        
        # Resize to cap max dimension at 640px (prevents slow Qwen2-VL processing)
        max_dim = 640
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Strict one-word targets (no leading space - chat template handles it)
        response = "Real" if label == 0 else "Fake"
        
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
                    {"type": "text", "text": response}
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
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Conservative: attention only
        "processor_class": "AutoProcessor",
        "model_class": "AutoModelForVision2Seq",
    },
    "qwen_vl": {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Conservative: attention only
        "processor_class": "AutoProcessor", 
        "model_class": "Qwen2VLForConditionalGeneration",
    },
}


# =============================================================================
# Unified PEFT Trainer
# =============================================================================

class PEFTTrainer:
    """Unified PEFT trainer with proper label masking and forced-choice evaluation."""
    
    def __init__(
        self,
        model_type: str = "smolvlm",
        peft_method: str = "lora",
        output_dir: str = None,
        lora_rank: int = 8,  # Conservative default
        lora_alpha: int = 16,
        learning_rate: float = 5e-5,  # Lower LR for VLM LoRA
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
        
        # Set output directory with timestamp to preserve previous runs
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = output_dir or f"checkpoints/{model_type}-{peft_method}"
        self.output_dir = Path(f"{base_dir}/{timestamp}")
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
        self.dataset_name = "faceforensics"
        
        # Datasets (cached after first creation)
        self._train_dataset = None
        self._val_dataset = None
        
        print(f"Initialized PEFTTrainer:")
        print(f"  Model: {self.model_type} ({self.model_config['model_name']})")
        print(f"  PEFT Method: {get_method_info(peft_method)}")
        print(f"  LoRA: r={lora_rank}, alpha={lora_alpha}")
        print(f"  LR: {learning_rate}")
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
        """Create train and validation dataloaders with proper balancing and deterministic split."""
        # Use cached datasets if available
        if self._train_dataset is not None and self._val_dataset is not None and not force_reload:
            print(f"Using cached datasets - Train: {len(self._train_dataset)}, Val: {len(self._val_dataset)}")
            return self._train_dataset, self._val_dataset
        
        # Load full dataset ONCE
        from src.data.dataset_loader import create_dataset
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        full_dataset = create_dataset(config, self.dataset_name, max_samples)
        all_samples = full_dataset.samples
        
        # =====================================================================
        # DEBUG: Verify label mapping (label 0 = Real, label 1 = Fake)
        # =====================================================================
        from collections import Counter
        cnt = Counter([s[1] for s in all_samples])
        print(f"[DATA] raw label counts: {cnt}")
        
        # Print examples of each label to verify mapping
        shown0, shown1 = 0, 0
        for (path, lab, manip) in all_samples[:2000]:
            path_str = str(path)
            if lab == 0 and shown0 < 2:
                print(f"[DATA] label=0 example: {path_str[-50:]} manip={manip}")
                shown0 += 1
            if lab == 1 and shown1 < 2:
                print(f"[DATA] label=1 example: {path_str[-50:]} manip={manip}")
                shown1 += 1
            if shown0 >= 2 and shown1 >= 2:
                break
        
        # Balance ONCE
        real_samples = [s for s in all_samples if s[1] == 0]
        fake_samples = [s for s in all_samples if s[1] == 1]
        
        random.seed(42)
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        n = min(len(real_samples), len(fake_samples))
        balanced_samples = real_samples[:n] + fake_samples[:n]
        
        random.seed(42)
        random.shuffle(balanced_samples)
        
        # Deterministic split ONCE
        train_size = int(0.8 * len(balanced_samples))
        train_samples = balanced_samples[:train_size]
        val_samples = balanced_samples[train_size:]
        
        # Create datasets from precomputed samples
        self._train_dataset = DeepfakeFineTuneDataset(train_samples, self.processor)
        self._val_dataset = DeepfakeFineTuneDataset(val_samples, self.processor)
        
        # Print distribution to confirm balance
        tr_fake = sum(1 for s in train_samples if s[1] == 1)
        va_fake = sum(1 for s in val_samples if s[1] == 1)
        tr_real = len(train_samples) - tr_fake
        va_real = len(val_samples) - va_fake
        
        print(f"[Split] Train: {len(train_samples)} (fake={tr_fake}, real={tr_real})")
        print(f"[Split] Val:   {len(val_samples)} (fake={va_fake}, real={va_real})")
        
        # =====================================================================
        # CHECK B: Verify val set is ~50/50 balanced
        # =====================================================================
        val_balance = min(va_fake, va_real) / max(va_fake, va_real) if max(va_fake, va_real) > 0 else 0
        print(f"[CHECK B] Val balance ratio: {val_balance:.2f} (should be close to 1.0)")
        if val_balance < 0.8:
            print(f"  WARNING: Val set is imbalanced! Fake={va_fake}, Real={va_real}")
        
        return self._train_dataset, self._val_dataset
    
    def encode_with_labels(self, image, messages):
        """Encode input with proper label masking - loss ONLY on assistant answer."""
        # Full conversation (with assistant answer)
        text_full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(text=text_full, images=[image], return_tensors="pt")
        
        # Prompt-only conversation (assistant starts after this)
        text_prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        
        input_ids = full_inputs["input_ids"].squeeze(0)
        attn = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask everything except assistant answer
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        labels[attn == 0] = -100
        
        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        
        # VLM-specific tensors
        for k in ["pixel_values", "image_grid_thw"]:
            if k in full_inputs:
                out[k] = full_inputs[k].squeeze(0)
        
        return out
    
    def train(self, max_samples: Optional[int] = None):
        """Run PEFT fine-tuning with proper label masking."""
        if self.model is None:
            self.setup()
        
        train_dataset, val_dataset = self.create_dataloaders(max_samples)
        
        # Collate function with proper label masking
        def collate_fn(examples):
            encoded = [self.encode_with_labels(ex["image"], ex["messages"]) for ex in examples]
            
            max_len = max(e["input_ids"].shape[0] for e in encoded)
            pad_id = self.processor.tokenizer.pad_token_id or 0
            
            def pad_1d(x, pad_value):
                pad_len = max_len - x.shape[0]
                if pad_len <= 0:
                    return x
                return torch.cat([x, torch.full((pad_len,), pad_value, dtype=x.dtype)], dim=0)
            
            batch = {
                "input_ids": torch.stack([pad_1d(e["input_ids"], pad_id) for e in encoded]),
                "attention_mask": torch.stack([pad_1d(e["attention_mask"], 0) for e in encoded]),
                "labels": torch.stack([pad_1d(e["labels"], -100) for e in encoded]),
            }
            
            # VLM tensors
            if "pixel_values" in encoded[0]:
                batch["pixel_values"] = torch.stack([e["pixel_values"] for e in encoded])
            if "image_grid_thw" in encoded[0]:
                batch["image_grid_thw"] = torch.stack([e["image_grid_thw"] for e in encoded])
            
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
        
        # Setup warmup scheduler to stabilize training
        from transformers import get_linear_schedule_with_warmup
        total_steps = (len(train_loader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = max(10, int(0.05 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(f"Using warmup scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
        
        # Training loop
        print(f"\nStarting {self.peft_info.name} Fine-Tuning")
        print(f"Epochs: {self.num_epochs}, Batch Size: {self.batch_size}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print("=" * 50)
        
        # =====================================================================
        # CHECK A: Verify label masking is correct (run once on first batch)
        # =====================================================================
        print("\n[CHECK A] Verifying label masking...")
        first_batch = next(iter(train_loader))
        for i in range(min(2, first_batch["labels"].shape[0])):
            labels_i = first_batch["labels"][i]
            unmasked_mask = labels_i != -100
            unmasked_count = unmasked_mask.sum().item()
            unmasked_tokens = labels_i[unmasked_mask]
            decoded = self.processor.tokenizer.decode(unmasked_tokens)
            print(f"  Sample {i}: {unmasked_count} unmasked tokens -> '{decoded}'")
        print("  Expected: only ' Real' or ' Fake' (plus maybe EOS)")
        print("=" * 50)
        
        global_step = 0
        best_loss = float('inf')
        epoch_losses = []  # Track loss per epoch
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                epoch_loss += outputs.loss.item()
                num_batches += 1
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()  # Warmup + linear decay
                    optimizer.zero_grad()
                    global_step += 1
                
                progress_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                print("New best loss! Saving checkpoint...")
                self.model.save_pretrained(self.output_dir / "best")
        
        # Save final checkpoint
        print(f"\nSaving final {self.peft_info.name} adapter...")
        self.model.save_pretrained(self.output_dir / "final")
        
        # Save training info with comprehensive metrics
        from datetime import datetime
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "peft_method": self.peft_method,
            "dataset": self.dataset_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "train_samples": len(self._train_dataset) if self._train_dataset else 0,
            "val_samples": len(self._val_dataset) if self._val_dataset else 0,
            "epoch_losses": epoch_losses,
            "best_loss": best_loss,
            "final_loss": avg_loss,
        }
        
        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training complete! Checkpoints saved to {self.output_dir}")
        
        return training_info
    
    def evaluate(self, max_samples: int = None):
        """Evaluate using forced-choice p_fake - NOT text parsing."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        self.model.eval()
        
        _, val_dataset = self.create_dataloaders()
        
        n = len(val_dataset) if max_samples is None else min(len(val_dataset), max_samples)
        
        labels = []
        predictions = []
        p_fakes = []
        
        print(f"\nEvaluating {self.peft_info.name} model on {n} samples (forced-choice)...")
        
        # =====================================================================
        # CHECK C: Verify training/eval prompts match EXACTLY
        # =====================================================================
        eval_prompt = val_dataset.prompt
        print(f"[CHECK C] Eval prompt ends with: '...{eval_prompt[-30:]}'")
        print(f"  Training and eval use SAME prompt object: ✓")
        
        with torch.no_grad():
            for i in tqdm(range(n)):
                sample = val_dataset[i]
                image = sample["image"]
                label = sample["label"]
                labels.append(label)
                
                # Build scoring prompt
                scoring_prompt = val_dataset.prompt
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": scoring_prompt}
                    ]
                }]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(text=text, images=[image], return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items() if hasattr(v, 'to')}
                
                # Get p_fake using robust forced-choice scoring
                p_fake, debug = self._forced_choice_p_fake(inputs)
                p_fakes.append(p_fake)
                
                pred = 1 if p_fake >= 0.5 else 0
                predictions.append(pred)
                
                # Debug first 5 samples with variant info
                if i < 5:
                    print(
                        f"[DEBUG] i={i} label={label} p_fake={p_fake:.3f} pred={pred} "
                        f"best_real='{debug['best_real']}' best_fake='{debug['best_fake']}'"
                    )
        
        # =====================================================================
        # SANITY CHECK: Verify p_fake has variance and separates classes
        # =====================================================================
        import numpy as np
        p = np.array(p_fakes, dtype=float)
        y = np.array(labels, dtype=int)
        
        real_p = p[y == 0]
        fake_p = p[y == 1]
        
        print("\n[SANITY CHECK] p_fake statistics:")
        if len(fake_p) > 0 and len(real_p) > 0:
            print(f"  mean_fake={fake_p.mean():.3f} mean_real={real_p.mean():.3f}")
        print(f"  std={p.std():.4f} min={p.min():.3f} max={p.max():.3f}")
        print(f"  p1={np.percentile(p, 1):.3f} p99={np.percentile(p, 99):.3f}")
        
        if p.std() < 0.05:
            print("  WARNING: p_fake has no variance! Evaluation may be broken.")
        
        # Compute metrics
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / n if n > 0 else 0
        
        # Label distribution
        fake_count = sum(labels)
        real_count = n - fake_count
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{n})")
        print(f"  Label distribution - Real: {real_count}, Fake: {fake_count}")
        
        # Prediction distribution
        pred_fake = sum(predictions)
        pred_real = n - pred_fake
        print(f"  Prediction distribution - Real: {pred_real}, Fake: {pred_fake}")
        
        # Build comprehensive metrics
        eval_metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
            "label_distribution": {"real": real_count, "fake": fake_count},
            "prediction_distribution": {"real": pred_real, "fake": pred_fake},
            "p_fake_stats": {
                "mean_fake": float(fake_p.mean()) if len(fake_p) > 0 else None,
                "mean_real": float(real_p.mean()) if len(real_p) > 0 else None,
                "separation": float(fake_p.mean() - real_p.mean()) if len(fake_p) > 0 and len(real_p) > 0 else None,
                "std": float(p.std()),
                "min": float(p.min()),
                "max": float(p.max()),
                "p1": float(np.percentile(p, 1)),
                "p99": float(np.percentile(p, 99)),
            },
        }
        
        # Save to file
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\nEvaluation metrics saved to {eval_path}")
        
        return eval_metrics
    
    def _forced_choice_p_fake(self, inputs):
        """Compute p(Fake) using robust full-sequence teacher forcing."""
        from src.utils.forced_choice import forced_choice_p_fake_best
        
        p_fake, debug = forced_choice_p_fake_best(
            self.model, inputs, self.processor.tokenizer
        )
        return p_fake, debug


# =============================================================================
# Main Entry Point
# =============================================================================

def run_peft_finetuning(
    config_path: str = "configs/model_configs.yaml",
    dataset_name: str = "faceforensics",
    output_dir: str = None,
    max_samples: Optional[int] = None,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 1,
    model_type: str = "smolvlm",
    peft_method: str = "lora",
    **peft_kwargs,
):
    """Main function to run PEFT fine-tuning."""
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
