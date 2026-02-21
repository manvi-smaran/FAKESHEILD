"""
CNN + PEFT Hybrid Fine-Tuning for Deepfake Detection

Combines a ForensicCNN (learns pixel-level manipulation artifacts) with
Qwen2-VL + LoRA (learns to use forensic evidence for classification).

Architecture:
  Image → ForensicCNN → forensic_tokens [B, 4, 1536]
  Image → Qwen2-VL Vision Encoder → image_tokens [B, ~256, 1536]
  Concat [forensic_tokens, image_tokens, text_tokens] → LLM + LoRA → "Real"/"Fake"

Training:
  Single optimizer trains CNN + projection + LoRA jointly end-to-end.
  CNN learns what forensic features help; LoRA learns to use them.
"""

import os
import json
import random
import yaml
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.models.forensic_cnn import ForensicCNN
from src.training.peft_configs import (
    PEFT_METHODS,
    create_peft_config,
    get_quantization_config,
    get_method_info,
)


# =============================================================================
# Model Configuration (same as peft_finetune.py)
# =============================================================================

MODEL_CONFIGS = {
    "qwen_vl": {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "hidden_size": 1536,
    },
}


# =============================================================================
# Dataset (same format, adds CNN preprocessing)
# =============================================================================

class CNNPEFTDataset(Dataset):
    """Dataset that returns both VLM-formatted data and raw images for CNN."""
    
    def __init__(self, samples: List[Tuple], processor, cnn_transform):
        self.samples = samples
        self.processor = processor
        self.cnn_transform = cnn_transform
        
        self.prompt = (
            "Classify authenticity.\n\n"
            "Answer with exactly one word: Real or Fake.\n\n"
            "Answer:"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label, manip_type = self.samples[idx]
        
        image = Image.open(frame_path).convert("RGB")
        
        # Resize for VLM (same as original peft_finetune.py)
        max_dim = 640
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # CNN preprocessed tensor (224x224 normalized)
        cnn_input = self.cnn_transform(image)
        
        response = "Real" if label == 0 else "Fake"
        
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
            "cnn_input": cnn_input,
            "messages": messages,
            "label": label,
        }


# =============================================================================
# CNN + PEFT Hybrid Trainer
# =============================================================================

class CNNPEFTTrainer:
    """
    Hybrid trainer that jointly trains ForensicCNN + LoRA adapters.
    
    The CNN extracts forensic features → projected as tokens → prepended
    to VLM input embeddings → LoRA adapters learn to use forensic evidence.
    """
    
    def __init__(
        self,
        model_type: str = "qwen_vl",
        peft_method: str = "lora",
        output_dir: str = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 5e-5,
        cnn_lr: float = 1e-4,
        num_epochs: int = 10,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        num_forensic_tokens: int = 4,
        **peft_kwargs,
    ):
        self.model_type = model_type
        self.peft_method = peft_method
        self.model_config = MODEL_CONFIGS[model_type]
        self.peft_info = PEFT_METHODS[peft_method]
        
        # Timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = output_dir or f"checkpoints/{model_type}-cnn-{peft_method}"
        self.output_dir = Path(f"{base_dir}/{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hyperparameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.cnn_lr = cnn_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_forensic_tokens = num_forensic_tokens
        self.peft_kwargs = peft_kwargs
        
        # Components
        self.model = None
        self.processor = None
        self.forensic_cnn = None
        self._train_dataset = None
        self._val_dataset = None
        
        # Config path (set externally)
        self.config_path = "configs/model_configs.yaml"
        self.dataset_name = "faceforensics"
        
        print("=" * 60)
        print("CNN + PEFT Hybrid Trainer")
        print("=" * 60)
        print(f"  Model: {model_type}")
        print(f"  PEFT: {get_method_info(peft_method)}")
        print(f"  LoRA: r={lora_rank}, alpha={lora_alpha}")
        print(f"  Forensic Tokens: {num_forensic_tokens}")
        print(f"  LR (LoRA): {learning_rate}, LR (CNN): {cnn_lr}")
        print(f"  Output: {self.output_dir}")
    
    def setup(self):
        """Load VLM + LoRA + ForensicCNN."""
        from transformers import AutoProcessor
        from peft import get_peft_model
        
        hidden_size = self.model_config["hidden_size"]
        
        # 1. Load processor
        print(f"\nLoading {self.model_type} processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_config["model_name"],
            trust_remote_code=True,
        )
        
        # 2. Load model
        print(f"Loading {self.model_type} model...")
        from transformers import Qwen2VLForConditionalGeneration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 3. Apply LoRA
        print(f"\nApplying {self.peft_info.name}...")
        peft_config = create_peft_config(
            peft_method=self.peft_method,
            target_modules=self.model_config["target_modules"],
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            **self.peft_kwargs,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # 4. Initialize ForensicCNN
        print(f"\nInitializing ForensicCNN ({self.num_forensic_tokens} tokens → {hidden_size}-dim)...")
        self.forensic_cnn = ForensicCNN(
            hidden_size=hidden_size,
            num_tokens=self.num_forensic_tokens,
        )
        
        # Move CNN to same device as model
        # Use float32 for CNN to avoid NaN from half-precision gradient accumulation
        device = next(self.model.parameters()).device
        self.forensic_cnn = self.forensic_cnn.to(device).to(torch.float32)
        
        total_cnn, trainable_cnn = self.forensic_cnn.count_parameters()
        print(f"  CNN parameters: {total_cnn:,} (all trainable)")
        
        # 5. Verify language_model sub-module is accessible for hook injection
        try:
            lm = self._get_language_model()
            print(f"  Language model found: {type(lm).__name__} (hook target ✓)")
        except RuntimeError as e:
            print(f"  [WARN] {e}")
            print("  Falling back — printing model structure for debugging:")
            for name, _ in self.model.named_modules():
                if name.count('.') <= 4:
                    print(f"    {name}")
        
        # Enable training
        self.model.train()
        self.forensic_cnn.train()
        
        # Ensure LoRA params require grad
        for name, param in self.model.named_parameters():
            if any(key in name.lower() for key in ['lora', 'prefix', 'adapter']):
                param.requires_grad = True
        
        print("Setup complete!")
    
    def create_dataloaders(self, max_samples=None):
        """Create balanced train/val splits (same logic as peft_finetune.py)."""
        if self._train_dataset is not None and self._val_dataset is not None:
            return self._train_dataset, self._val_dataset
        
        from src.data.dataset_loader import create_dataset
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        full_dataset = create_dataset(config, self.dataset_name, max_samples)
        all_samples = full_dataset.samples
        
        from collections import Counter
        cnt = Counter([s[1] for s in all_samples])
        print(f"[DATA] raw label counts: {cnt}")
        
        # Balance
        real_samples = [s for s in all_samples if s[1] == 0]
        fake_samples = [s for s in all_samples if s[1] == 1]
        
        random.seed(42)
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        n = min(len(real_samples), len(fake_samples))
        balanced_samples = real_samples[:n] + fake_samples[:n]
        
        random.seed(42)
        random.shuffle(balanced_samples)
        
        train_size = int(0.8 * len(balanced_samples))
        train_samples = balanced_samples[:train_size]
        val_samples = balanced_samples[train_size:]
        
        print(f"[DATA] Balanced: {n} per class = {2*n} total")
        print(f"[DATA] Train: {len(train_samples)}, Val: {len(val_samples)}")
        
        self._train_dataset = CNNPEFTDataset(
            train_samples, self.processor, self.forensic_cnn.transform
        )
        self._val_dataset = CNNPEFTDataset(
            val_samples, self.processor, self.forensic_cnn.transform
        )
        
        return self._train_dataset, self._val_dataset
    
    def encode_with_labels(self, image, messages):
        """Encode a sample with proper label masking (same as peft_finetune.py)."""
        text_full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(text=text_full, images=[image], return_tensors="pt")
        
        text_prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        
        input_ids = full_inputs["input_ids"].squeeze(0)
        attn = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        labels[attn == 0] = -100
        
        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        
        for k in ["pixel_values", "image_grid_thw"]:
            if k in full_inputs:
                out[k] = full_inputs[k].squeeze(0)
        
        return out
    
    def _find_embed_tokens(self):
        """Find the embed_tokens module using standard HuggingFace API."""
        embed = self.model.get_input_embeddings()
        if embed is not None:
            return embed
        raise RuntimeError("Cannot find embed_tokens module in model")
    
    def _get_language_model(self):
        """
        Find the inner language_model sub-module (the transformer decoder)
        through the PEFT wrapper hierarchy.
        
        In Qwen2-VL: Qwen2VLModel contains language_model + visual encoder.
        After PEFT: PeftModel → base_model → model → model → language_model
        We need the language_model to register our injection hook.
        """
        # Search named_modules for the language_model that contains transformer layers
        for name, module in self.model.named_modules():
            if name.endswith('.language_model') and hasattr(module, 'layers'):
                return module
        
        # Fallback: try direct attribute paths
        candidates = [
            lambda: self.model.base_model.model.model.language_model,
            lambda: self.model.model.model.language_model,
            lambda: self.model.base_model.model.model,  # older structure without language_model
        ]
        for get_lm in candidates:
            try:
                lm = get_lm()
                if lm is not None and hasattr(lm, 'layers'):
                    return lm
            except AttributeError:
                continue
        
        raise RuntimeError(
            "Cannot find language_model sub-module. "
            "Please check model structure with model.named_modules()"
        )
    
    def _forward_with_forensic_tokens(self, batch, forensic_tokens):
        """
        Run model forward with forensic tokens injected via hook.
        
        Strategy (hook-based injection):
        1. Pass input_ids + pixel_values normally to the model
        2. Qwen2-VL internally: embeds text → merges vision tokens → computes 3D RoPE
        3. A pre-hook on language_model intercepts inputs_embeds AFTER vision merging
        4. Hook prepends forensic tokens and extends attention_mask
        5. LLM decoder processes [forensic_tokens, vision+text_tokens] correctly
        
        This ensures vision processing and position IDs are computed correctly
        before forensic tokens are injected.
        
        Args:
            batch: dict with input_ids, attention_mask, labels, pixel_values, etc.
            forensic_tokens: [B, num_tokens, hidden_size] from ForensicCNN
            
        Returns:
            Model outputs with loss
        """
        device = forensic_tokens.device
        B = forensic_tokens.shape[0]
        num_ft = self.num_forensic_tokens
        
        # Pre-pad labels with -100 for forensic token positions
        # (labels are consumed by the outer forward, not the language_model)
        forensic_labels = torch.full((B, num_ft), -100, dtype=batch["labels"].dtype, device=device)
        padded_labels = torch.cat([forensic_labels, batch["labels"]], dim=1)
        
        # Get the language_model sub-module for hook registration
        language_model = self._get_language_model()
        
        # Storage for the hook to access forensic tokens
        _hook_state = {"forensic_tokens": forensic_tokens, "consumed": False}
        
        def inject_forensic_hook(module, args, kwargs):
            """Pre-hook that prepends forensic tokens to inputs_embeds."""
            if _hook_state["consumed"]:
                return args, kwargs
            
            ft = _hook_state["forensic_tokens"]
            _hook_state["consumed"] = True
            
            # Extract inputs_embeds from kwargs or args
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is None and len(args) > 0:
                # inputs_embeds might be a positional arg
                return args, kwargs
            
            if inputs_embeds is None:
                return args, kwargs
            
            # Cast forensic tokens to match embedding dtype
            ft = ft.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            
            # Prepend forensic tokens
            kwargs["inputs_embeds"] = torch.cat([ft, inputs_embeds], dim=1)
            
            # Extend attention_mask if present
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is not None:
                forensic_attn = torch.ones(
                    B, num_ft, dtype=attn_mask.dtype, device=attn_mask.device
                )
                kwargs["attention_mask"] = torch.cat([forensic_attn, attn_mask], dim=1)
            
            # Extend position_ids if present (simple sequential for forensic tokens)
            position_ids = kwargs.get("position_ids", None)
            if position_ids is not None and position_ids.dim() == 3:
                # Qwen2-VL uses 3D position_ids [3, B, seq_len]
                # For forensic tokens, use simple sequential positions starting at 0
                ft_pos = torch.zeros(
                    3, B, num_ft, dtype=position_ids.dtype, device=position_ids.device
                )
                for i in range(num_ft):
                    ft_pos[:, :, i] = i
                # Shift original position_ids by num_ft
                kwargs["position_ids"] = torch.cat(
                    [ft_pos, position_ids + num_ft], dim=2
                )
            elif position_ids is not None and position_ids.dim() == 2:
                # Standard 2D position_ids [B, seq_len]
                ft_pos = torch.arange(
                    num_ft, dtype=position_ids.dtype, device=position_ids.device
                ).unsqueeze(0).expand(B, -1)
                kwargs["position_ids"] = torch.cat(
                    [ft_pos, position_ids + num_ft], dim=1
                )
            
            # Extend cache_position if present
            cache_position = kwargs.get("cache_position", None)
            if cache_position is not None:
                ft_cache_pos = torch.arange(
                    num_ft, dtype=cache_position.dtype, device=cache_position.device
                )
                kwargs["cache_position"] = torch.cat(
                    [ft_cache_pos, cache_position + num_ft], dim=0
                )
            
            return args, kwargs
        
        # Register the hook (with_kwargs=True to access kwargs)
        hook_handle = language_model.register_forward_pre_hook(
            inject_forensic_hook, with_kwargs=True
        )
        
        try:
            # Build forward kwargs — pass input_ids + pixel_values normally
            # Let Qwen2-VL handle vision processing internally
            forward_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": padded_labels,
            }
            
            # Include vision inputs for Qwen2-VL's internal processing
            if "pixel_values" in batch:
                forward_kwargs["pixel_values"] = batch["pixel_values"]
            if "image_grid_thw" in batch:
                forward_kwargs["image_grid_thw"] = batch["image_grid_thw"]
            
            # Forward pass — model handles embedding + vision merge → hook injects forensic tokens → LLM decoder
            outputs = self.model(**forward_kwargs)
        finally:
            # Always remove hook to prevent accumulation
            hook_handle.remove()
        
        return outputs
    
    def train(self, max_samples=None):
        """Run CNN + PEFT hybrid fine-tuning."""
        if self.model is None:
            self.setup()
        
        train_dataset, val_dataset = self.create_dataloaders(max_samples)
        
        # Collate function
        def collate_fn(examples):
            encoded = [self.encode_with_labels(ex["image"], ex["messages"]) for ex in examples]
            cnn_inputs = torch.stack([ex["cnn_input"] for ex in examples])
            
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
                "cnn_inputs": cnn_inputs,
            }
            
            if "pixel_values" in encoded[0]:
                batch["pixel_values"] = torch.stack([e["pixel_values"] for e in encoded])
            if "image_grid_thw" in encoded[0]:
                batch["image_grid_thw"] = torch.stack([e["image_grid_thw"] for e in encoded])
            
            return batch
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
        
        # =====================================================================
        # Setup optimizer with different LRs for CNN vs LoRA
        # =====================================================================
        optimizer = torch.optim.AdamW([
            # LoRA parameters (lower LR)
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad],
             "lr": self.learning_rate},
            # CNN parameters (higher LR since training from scratch)
            {"params": self.forensic_cnn.parameters(),
             "lr": self.cnn_lr},
        ], weight_decay=0.01)
        
        # Warmup scheduler
        from transformers import get_linear_schedule_with_warmup
        total_steps = (len(train_loader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = max(10, int(0.05 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # =====================================================================
        # Training Loop
        # =====================================================================
        print(f"\nStarting CNN + {self.peft_info.name} Hybrid Training")
        print(f"Epochs: {self.num_epochs}, Batch Size: {self.batch_size}")
        print(f"Forensic Tokens: {self.num_forensic_tokens}")
        print("=" * 50)
        
        global_step = 0
        best_loss = float('inf')
        epoch_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            self.model.train()
            self.forensic_cnn.train()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                device = next(self.model.parameters()).device
                
                # 1. Extract forensic features with CNN (CNN runs in float32 for stability)
                cnn_inputs = batch.pop("cnn_inputs").to(device).to(torch.float32)
                forensic_tokens = self.forensic_cnn(cnn_inputs)  # [B, 4, 1536] in float32
                
                # 2. Move VLM inputs to device
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # 3. Forward pass with hook-based forensic token injection
                # The model processes input_ids + pixel_values normally
                # (vision merging, 3D RoPE, etc. all happen correctly)
                # A pre-hook on language_model prepends forensic tokens
                # right before the LLM decoder processes them
                outputs = self._forward_with_forensic_tokens(batch, forensic_tokens)
                
                # NaN-safe loss handling
                raw_loss = outputs.loss
                if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                    print(f"[WARN] Step {step}: NaN/Inf loss detected, skipping backward")
                    optimizer.zero_grad()
                    num_batches += 1
                    continue
                
                loss = raw_loss / self.gradient_accumulation_steps
                
                # Backward (gradients flow to both LoRA and CNN)
                loss.backward()
                
                epoch_loss += raw_loss.item()
                num_batches += 1
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients for both LoRA and CNN
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.forensic_cnn.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                progress_bar.set_postfix({"loss": f"{raw_loss.item():.4f}"})
                
                # Debug: log first few steps
                if step < 3 and epoch == 0:
                    print(f"  [DEBUG] step={step} loss={raw_loss.item():.6f} "
                          f"forensic_norm={forensic_tokens.norm():.4f}")
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print("New best! Saving checkpoint...")
                self.model.save_pretrained(self.output_dir / "best")
                torch.save(self.forensic_cnn.state_dict(), self.output_dir / "best" / "forensic_cnn.pt")
        
        # Save final
        self.model.save_pretrained(self.output_dir / "final")
        torch.save(self.forensic_cnn.state_dict(), self.output_dir / "final" / "forensic_cnn.pt")
        
        # Save training info
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "approach": "cnn_peft_hybrid",
            "model_type": self.model_type,
            "peft_method": self.peft_method,
            "dataset": self.dataset_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "cnn_lr": self.cnn_lr,
            "num_forensic_tokens": self.num_forensic_tokens,
            "epoch_losses": epoch_losses,
            "best_loss": best_loss,
            "final_loss": avg_loss,
        }
        
        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\nTraining complete! Saved to {self.output_dir}")
        return training_info
    
    def evaluate(self, max_samples=None):
        """Evaluate using forced-choice p_fake scoring WITH forensic token injection."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        self.model.eval()
        self.forensic_cnn.eval()
        
        _, val_dataset = self.create_dataloaders()
        n = len(val_dataset) if max_samples is None else min(len(val_dataset), max_samples)
        
        labels_list = []
        predictions = []
        p_fakes = []
        
        print(f"\nEvaluating CNN + {self.peft_info.name} on {n} samples...")
        print(f"  (forensic tokens ENABLED via hook injection)")
        
        # Get language_model for hook registration
        language_model = self._get_language_model()
        
        # Shared state for the hook — updated per sample
        _eval_hook_state = {"forensic_tokens": None, "active": False}
        
        def eval_inject_hook(module, args, kwargs):
            """Persistent hook that injects forensic tokens during eval."""
            if not _eval_hook_state["active"] or _eval_hook_state["forensic_tokens"] is None:
                return args, kwargs
            
            ft = _eval_hook_state["forensic_tokens"]
            inputs_embeds = kwargs.get("inputs_embeds", None)
            
            if inputs_embeds is None:
                return args, kwargs
            
            B = inputs_embeds.shape[0]
            num_ft = ft.shape[1]
            
            # Cast and prepend
            ft = ft.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            
            # Handle batch size mismatch (ft is [1, num_ft, hidden], embeds may differ)
            if ft.shape[0] != B:
                ft = ft.expand(B, -1, -1)
            
            kwargs["inputs_embeds"] = torch.cat([ft, inputs_embeds], dim=1)
            
            # Extend attention_mask
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is not None:
                forensic_attn = torch.ones(
                    B, num_ft, dtype=attn_mask.dtype, device=attn_mask.device
                )
                kwargs["attention_mask"] = torch.cat([forensic_attn, attn_mask], dim=1)
            
            # Extend position_ids
            position_ids = kwargs.get("position_ids", None)
            if position_ids is not None and position_ids.dim() == 3:
                ft_pos = torch.zeros(
                    3, B, num_ft, dtype=position_ids.dtype, device=position_ids.device
                )
                for i in range(num_ft):
                    ft_pos[:, :, i] = i
                kwargs["position_ids"] = torch.cat(
                    [ft_pos, position_ids + num_ft], dim=2
                )
            elif position_ids is not None and position_ids.dim() == 2:
                ft_pos = torch.arange(
                    num_ft, dtype=position_ids.dtype, device=position_ids.device
                ).unsqueeze(0).expand(B, -1)
                kwargs["position_ids"] = torch.cat(
                    [ft_pos, position_ids + num_ft], dim=1
                )
            
            return args, kwargs
        
        # Register persistent hook for entire evaluation
        hook_handle = language_model.register_forward_pre_hook(
            eval_inject_hook, with_kwargs=True
        )
        
        try:
            with torch.no_grad():
                for i in tqdm(range(n)):
                    sample = val_dataset[i]
                    image = sample["image"]
                    cnn_input = sample["cnn_input"]
                    label = sample["label"]
                    labels_list.append(label)
                    
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
                    
                    # Compute forensic tokens and activate hook
                    device = next(self.model.parameters()).device
                    cnn_tensor = cnn_input.unsqueeze(0).to(device).to(torch.float32)
                    forensic_tokens = self.forensic_cnn(cnn_tensor)
                    
                    # Update hook state — all model forward calls will now include forensic tokens
                    _eval_hook_state["forensic_tokens"] = forensic_tokens
                    _eval_hook_state["active"] = True
                    
                    # Forced-choice scoring (model forwards now include forensic tokens via hook)
                    from src.utils.forced_choice import forced_choice_p_fake_best
                    p_fake, debug = forced_choice_p_fake_best(
                        self.model, inputs, self.processor.tokenizer
                    )
                    
                    # Deactivate hook between samples
                    _eval_hook_state["active"] = False
                    
                    p_fakes.append(p_fake)
                    pred = 1 if p_fake >= 0.5 else 0
                    predictions.append(pred)
                    
                    if i < 5:
                        print(f"[DEBUG] i={i} label={label} p_fake={p_fake:.3f} pred={pred}")
        finally:
            hook_handle.remove()
        
        # Compute metrics
        import numpy as np
        p = np.array(p_fakes, dtype=float)
        y = np.array(labels_list, dtype=int)
        
        real_p = p[y == 0]
        fake_p = p[y == 1]
        
        correct = sum(1 for pr, l in zip(predictions, labels_list) if pr == l)
        accuracy = correct / n if n > 0 else 0
        
        print(f"\n[SANITY CHECK] p_fake statistics:")
        if len(fake_p) > 0 and len(real_p) > 0:
            print(f"  mean_fake={fake_p.mean():.3f} mean_real={real_p.mean():.3f}")
        print(f"  std={p.std():.4f} min={p.min():.3f} max={p.max():.3f}")
        
        fake_count = sum(labels_list)
        real_count = n - fake_count
        pred_fake = sum(predictions)
        pred_real = n - pred_fake
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{n})")
        print(f"  Label distribution - Real: {real_count}, Fake: {fake_count}")
        print(f"  Prediction distribution - Real: {pred_real}, Fake: {pred_fake}")
        
        eval_metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
            "approach": "cnn_peft_hybrid",
            "label_distribution": {"real": real_count, "fake": fake_count},
            "prediction_distribution": {"real": pred_real, "fake": pred_fake},
            "p_fake_stats": {
                "mean_fake": float(fake_p.mean()) if len(fake_p) > 0 else None,
                "mean_real": float(real_p.mean()) if len(real_p) > 0 else None,
                "separation": float(fake_p.mean() - real_p.mean()) if len(fake_p) > 0 and len(real_p) > 0 else None,
                "std": float(p.std()),
                "min": float(p.min()),
                "max": float(p.max()),
            },
        }
        
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\nMetrics saved to {eval_path}")
        
        return eval_metrics


# =============================================================================
# Main Entry Point
# =============================================================================

def run_cnn_peft_finetuning(
    config_path="configs/model_configs.yaml",
    dataset_name="faceforensics",
    output_dir=None,
    max_samples=None,
    num_epochs=10,
    learning_rate=5e-5,
    cnn_lr=1e-4,
    lora_rank=8,
    lora_alpha=16,
    batch_size=1,
    model_type="qwen_vl",
    peft_method="lora",
    num_forensic_tokens=4,
    **peft_kwargs,
):
    """Main function to run CNN + PEFT hybrid fine-tuning."""
    trainer = CNNPEFTTrainer(
        model_type=model_type,
        peft_method=peft_method,
        output_dir=output_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        cnn_lr=cnn_lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_forensic_tokens=num_forensic_tokens,
        **peft_kwargs,
    )
    
    trainer.config_path = config_path
    trainer.dataset_name = dataset_name
    
    training_info = trainer.train(max_samples=max_samples)
    metrics = trainer.evaluate()
    
    return trainer, {**training_info, **metrics}
