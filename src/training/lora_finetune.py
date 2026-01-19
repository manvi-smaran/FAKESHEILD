"""
LoRA Fine-Tuning Module for SmolVLM Deepfake Detection
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import yaml


class DeepfakeFineTuneDataset(Dataset):
    """Dataset for LoRA fine-tuning on deepfake detection."""
    
    def __init__(
        self,
        data_dir: str,
        processor,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        
        # Load samples
        self.samples = self._load_samples(max_samples)
        
        # Prompt template for training
        self.prompt = """Analyze this image of a human face carefully.
Is this a real photograph or a deepfake/AI-manipulated image?
Answer with 'Real' or 'Fake' followed by a brief explanation."""
    
    def _load_samples(self, max_samples: Optional[int]) -> List[Dict]:
        """Load image paths and labels."""
        samples = []
        
        # Load real images
        real_dir = self.data_dir / "Celeb-real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                samples.append({
                    "image_path": str(img_path),
                    "label": 0,
                    "label_text": "Real",
                    "response": "Real. This appears to be an authentic photograph with natural skin texture, consistent lighting, and no visible manipulation artifacts."
                })
        
        # Load fake images
        fake_dir = self.data_dir / "Celeb-synthesis"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                samples.append({
                    "image_path": str(img_path),
                    "label": 1,
                    "label_text": "Fake",
                    "response": "Fake. This image shows signs of deepfake manipulation including subtle blending artifacts around facial boundaries and inconsistencies in texture."
                })
        
        # Shuffle and limit
        import random
        random.shuffle(samples)
        
        if max_samples:
            samples = samples[:max_samples]
        
        # Split train/val (80/20)
        split_idx = int(len(samples) * 0.8)
        if self.split == "train":
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Create conversation format for SmolVLM
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
                    {"type": "text", "text": sample["response"]}
                ]
            }
        ]
        
        return {
            "image": image,
            "messages": messages,
            "label": sample["label"],
        }


class LoRATrainer:
    """LoRA fine-tuning trainer for SmolVLM."""
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        output_dir: str = "checkpoints/smolvlm-lora",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    def setup(self):
        """Load model and apply LoRA."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from peft import LoraConfig, get_peft_model, TaskType
        
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        print("Loading model...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Configure LoRA
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("LoRA setup complete!")
    
    def create_dataloaders(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
    ):
        """Create train and validation dataloaders."""
        train_dataset = DeepfakeFineTuneDataset(
            data_dir=data_dir,
            processor=self.processor,
            split="train",
            max_samples=max_samples,
        )
        
        val_dataset = DeepfakeFineTuneDataset(
            data_dir=data_dir,
            processor=self.processor,
            split="val",
            max_samples=max_samples,
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
    ):
        """Run LoRA fine-tuning."""
        from transformers import TrainingArguments, Trainer
        
        if self.model is None:
            self.setup()
        
        train_dataset, val_dataset = self.create_dataloaders(data_dir, max_samples)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Custom data collator
        def collate_fn(examples):
            images = [ex["image"] for ex in examples]
            messages_list = [ex["messages"] for ex in examples]
            labels = [ex["label"] for ex in examples]
            
            # Process each example
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_labels = []
            
            for image, messages in zip(images, messages_list):
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                )
                
                # Process inputs
                inputs = self.processor(
                    text=text,
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                
                batch_input_ids.append(inputs["input_ids"].squeeze(0))
                batch_attention_mask.append(inputs["attention_mask"].squeeze(0))
                if "pixel_values" in inputs:
                    batch_pixel_values.append(inputs["pixel_values"].squeeze(0))
            
            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),  # For causal LM
            }
            
            if batch_pixel_values:
                result["pixel_values"] = torch.stack(batch_pixel_values)
            
            return result
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
        )
        
        print("\n" + "="*60)
        print("Starting LoRA Fine-Tuning")
        print("="*60)
        
        # Train
        trainer.train()
        
        # Save final model
        print("\nSaving LoRA adapter...")
        self.model.save_pretrained(self.output_dir / "final")
        self.processor.save_pretrained(self.output_dir / "final")
        
        print(f"\nTraining complete! Model saved to: {self.output_dir / 'final'}")
        
        return trainer
    
    def evaluate(self, data_dir: str, max_samples: int = 100):
        """Evaluate the fine-tuned model."""
        from src.evaluation.metrics import parse_prediction, compute_metrics
        
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() or train() first.")
        
        val_dataset = DeepfakeFineTuneDataset(
            data_dir=data_dir,
            processor=self.processor,
            split="val",
            max_samples=max_samples,
        )
        
        predictions = []
        labels = []
        
        self.model.eval()
        
        prompt = """Analyze this image of a human face carefully.
Is this a real photograph or a deepfake/AI-manipulated image?
Answer with 'Real' or 'Fake'."""
        
        for sample in tqdm(val_dataset, desc="Evaluating"):
            image = sample["image"]
            label = sample["label"]
            
            # Create input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            
            inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )
            
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            pred = parse_prediction(response)
            predictions.append(pred)
            labels.append(label)
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, [0.5] * len(predictions))
        
        print("\n" + "="*60)
        print("Fine-Tuned Model Evaluation Results")
        print("="*60)
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("="*60)
        
        return metrics


def run_lora_finetuning(
    data_dir: str = "./data/celeb_df",
    output_dir: str = "checkpoints/smolvlm-lora",
    max_samples: Optional[int] = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
):
    """Main function to run LoRA fine-tuning."""
    trainer = LoRATrainer(
        output_dir=output_dir,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    
    trainer.train(data_dir=data_dir, max_samples=max_samples)
    
    # Evaluate
    metrics = trainer.evaluate(data_dir=data_dir)
    
    return trainer, metrics
