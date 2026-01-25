from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import yaml

from src.models import get_model
from src.data.dataset_loader import create_dataset, DeepfakeDataset
from src.utils.prompts import (
    build_few_shot_prompt, 
    FEW_SHOT_EXAMPLE_REAL, 
    FEW_SHOT_EXAMPLE_FAKE,
    JSON_FEW_SHOT_SYSTEM,
    JSON_FEW_SHOT_EXAMPLE_REAL,
    JSON_FEW_SHOT_EXAMPLE_FAKE,
    JSON_FEW_SHOT_QUERY,
    parse_json_response,
)
from src.evaluation.metrics import (
    parse_prediction,
    extract_confidence,
    compute_metrics,
    compute_per_manipulation_metrics,
    format_results,
)


class FewShotEvaluator:
    
    def __init__(
        self,
        config_path: str = "configs/model_configs.yaml",
        results_dir: str = "results",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_few_shot_examples(
        self, 
        dataset: DeepfakeDataset, 
        k: int,
    ) -> Tuple[List[Tuple[Image.Image, int, str]], List[int]]:
        """Get stratified examples and their actual dataset indices."""
        examples, example_indices = dataset.get_stratified_samples(
            n_per_class=k // 2, 
            return_indices=True
        )
        return examples, example_indices
    
    def _build_few_shot_context(self, examples: List[Tuple[Image.Image, int, str]]) -> str:
        example_texts = []
        
        for idx, (image, label, manip_type) in enumerate(examples, 1):
            if label == 0:
                example_texts.append(FEW_SHOT_EXAMPLE_REAL.format(idx=idx))
            else:
                example_texts.append(FEW_SHOT_EXAMPLE_FAKE.format(idx=idx))
        
        return "\n\n".join(example_texts)
    
    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        k_values: List[int] = None,
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        if k_values is None:
            k_values = self.config["evaluation"]["few_shot_k"]
        
        model_config = self.config["models"][model_name]
        model = get_model(model_name, model_config)
        
        print(f"Loading model: {model_config['name']}...")
        model.load_model()
        
        dataset = create_dataset(self.config, dataset_name, max_samples)
        print(f"Dataset loaded: {len(dataset)} samples")
        
        results_by_k = {}
        
        for k in k_values:
            print(f"\n--- Few-Shot with k={k} ---")
            
            examples, example_indices = self._get_few_shot_examples(dataset, k)
            few_shot_context = self._build_few_shot_context(examples)
            
            prompt = f"""You are analyzing facial images to detect deepfakes. Here are some examples:

{few_shot_context}

Now analyze this new image:
Is this a real photograph or a deepfake? Answer 'Real' or 'Fake' with your reasoning."""
            
            predictions = []
            confidences = []
            labels = []
            manipulation_types = []
            
            # Convert to set for O(1) lookup
            example_indices_set = set(example_indices)
            
            for idx, (image, label, manip_type) in enumerate(tqdm(dataset, desc=f"k={k}")):
                if idx in example_indices_set:
                    continue
                
                response = model.predict(image, prompt)
                
                pred = parse_prediction(response)
                conf = extract_confidence(response)
                
                predictions.append(pred)
                confidences.append(conf)
                labels.append(label)
                manipulation_types.append(manip_type)
            
            metrics = compute_metrics(predictions, labels, confidences)
            per_manip_metrics = compute_per_manipulation_metrics(
                predictions, labels, manipulation_types
            )
            
            results_by_k[f"k={k}"] = {
                "k": k,
                "metrics": metrics,
                "per_manipulation": per_manip_metrics,
                "num_samples": len(predictions),
            }
            
            print(format_results(metrics, model_config["name"], dataset_name, f"Few-Shot (k={k})"))
        
        model.cleanup()
        
        results = {
            "model": model_name,
            "model_full_name": model_config["name"],
            "dataset": dataset_name,
            "results_by_k": results_by_k,
        }
        
        return results
    
    def evaluate_model_json(
        self,
        model_name: str,
        dataset_name: str,
        k_values: List[int] = None,
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        JSON-based few-shot evaluation with actual images.
        Returns structured predictions including p_fake probabilities.
        """
        if k_values is None:
            k_values = self.config["evaluation"]["few_shot_k"]
        
        model_config = self.config["models"][model_name]
        model = get_model(model_name, model_config)
        
        print(f"Loading model: {model_config['name']}...")
        model.load_model()
        
        dataset = create_dataset(self.config, dataset_name, max_samples)
        print(f"Dataset loaded: {len(dataset)} samples")
        
        results_by_k = {}
        
        for k in k_values:
            print(f"\n--- JSON Few-Shot with k={k} ---")
            
            # Get stratified examples with actual images
            examples, example_indices = self._get_few_shot_examples(dataset, k)
            
            # Build example texts for prompt
            example_texts = []
            example_images = []
            for idx, (img, label, _) in enumerate(examples, 1):
                if label == 0:
                    example_texts.append(JSON_FEW_SHOT_EXAMPLE_REAL.format(idx=idx))
                else:
                    example_texts.append(JSON_FEW_SHOT_EXAMPLE_FAKE.format(idx=idx))
                example_images.append(img)
            
            # Build TWO prompts: one for multi-image, one for single-image fallback
            # Multi-image prompt: EXPLICIT ordering contract for image binding
            multi_image_prompt = f"""{JSON_FEW_SHOT_SYSTEM}

IMAGE ORDER: Images are provided in sequence - Examples 1 through {k}, then the Query image last.
Analyze the LAST image (Query) based on patterns learned from the {k} example images.

EXAMPLES:
{chr(10).join(example_texts)}

{JSON_FEW_SHOT_QUERY}"""
            
            # Single-image fallback prompt: text-only exemplars, no claim about shown images
            single_image_prompt = f"""{JSON_FEW_SHOT_SYSTEM}

REFERENCE EXAMPLES (text descriptions only - no images provided for these):
{chr(10).join(example_texts)}

Based on these example patterns, analyze the image and output ONLY JSON:"""
            
            predictions = []
            confidences = []  # p_fake scores (None for parse failures)
            labels = []
            manipulation_types = []
            all_evidence = []
            parse_successes = []  # Track successful parses
            
            # Convert to set for O(1) lookup
            example_indices_set = set(example_indices)
            
            for idx, (image, label, manip_type) in enumerate(tqdm(dataset, desc=f"k={k} JSON")):
                if idx in example_indices_set:
                    continue
                
                # Try multi-image predict if available, else single
                # Use broad exception handling for robustness (Issue #4)
                try:
                    all_images = example_images + [image]
                    response = model.predict_multi_image(all_images, multi_image_prompt)
                except Exception as e:
                    # Fallback to single image with text-only prompt
                    # Catches AttributeError, RuntimeError, OOM, API errors, etc.
                    try:
                        response = model.predict(image, single_image_prompt)
                    except Exception as e2:
                        # Complete failure - skip this sample
                        predictions.append(-1)
                        confidences.append(None)
                        labels.append(label)
                        manipulation_types.append(manip_type)
                        all_evidence.append([f"model error: {str(e2)[:50]}"])
                        parse_successes.append(False)
                        continue
                
                # Parse JSON response
                parsed = parse_json_response(response)
                
                # Handle parse failures - use pred=-1 for unknown
                if parsed["label"] == "unknown":
                    pred = -1
                else:
                    pred = 1 if parsed["label"] == "fake" else 0
                
                p_fake = parsed["p_fake"]  # May be None for parse failures
                evidence = parsed["evidence"]
                
                predictions.append(pred)
                confidences.append(p_fake)
                labels.append(label)
                manipulation_types.append(manip_type)
                all_evidence.append(evidence)
                parse_successes.append(parsed.get("parse_success", False))
            
            # Filter out None confidences for AUC calculation
            valid_conf = [c for c in confidences if c is not None]
            valid_conf_mask = [c is not None for c in confidences]
            
            # Compute metrics: Classification on ALL valid preds, AUC on None-filtered subset
            # compute_metrics now handles this internally with auc_valid_ratio
            metrics = compute_metrics(predictions, labels, confidences, scores_are_proba=True)
            
            # Add parse success ratio
            metrics["parse_success_ratio"] = sum(parse_successes) / len(parse_successes) if parse_successes else 0.0
            metrics["auc_valid_samples"] = len(valid_conf)
            
            # Per-manipulation metrics with p_fake (compute_metrics handles None filtering internally)
            per_manip_metrics = compute_per_manipulation_metrics(
                predictions, labels, manipulation_types, confidences, scores_are_proba=True
            )
            
            results_by_k[f"k={k}"] = {
                "k": k,
                "metrics": metrics,
                "per_manipulation": per_manip_metrics,
                "num_samples": len(predictions),
                "sample_evidence": all_evidence[:10],  # Save sample evidence
            }
            
            print(format_results(metrics, model_config["name"], dataset_name, f"JSON Few-Shot (k={k})"))
        
        model.cleanup()
        
        results = {
            "model": model_name,
            "model_full_name": model_config["name"],
            "dataset": dataset_name,
            "evaluation_type": "json_few_shot",
            "results_by_k": results_by_k,
        }
        
        if save_predictions:
            # Include model_name in filename to prevent overwrites
            self._save_results(results, dataset_name, f"json_few_shot_{model_name}")
        
        return results
    
    def evaluate_all_models(
        self,
        dataset_name: str,
        k_values: List[int] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        all_results = {}
        
        for model_name in self.config["models"].keys():
            print(f"\n{'#'*60}")
            print(f"# Evaluating: {model_name}")
            print(f"{'#'*60}")
            
            try:
                results = self.evaluate_model(
                    model_name, 
                    dataset_name, 
                    k_values,
                    max_samples,
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        
        self._save_results(all_results, dataset_name, "few_shot")
        
        return all_results
    
    def _save_results(
        self, 
        results: Dict[str, Any], 
        dataset_name: str,
        eval_type: str,
    ):
        output_path = self.results_dir / f"{eval_type}_{dataset_name}_results.json"
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def run_few_shot_evaluation(
    model_name: Optional[str] = None,
    dataset_name: str = "celebdf",
    k_values: List[int] = None,
    max_samples: Optional[int] = None,
    config_path: str = "configs/model_configs.yaml",
    use_json: bool = False,
):
    """
    Run few-shot evaluation.
    
    Args:
        model_name: Specific model to evaluate, or None for all models
        dataset_name: Dataset to evaluate on
        k_values: List of k values for few-shot
        max_samples: Maximum samples to evaluate
        config_path: Path to config file
        use_json: If True, use JSON-based evaluation with structured output
    """
    evaluator = FewShotEvaluator(config_path=config_path)
    
    if use_json:
        if model_name:
            return evaluator.evaluate_model_json(model_name, dataset_name, k_values, max_samples)
        else:
            # Evaluate all models with JSON
            all_results = {}
            for name in evaluator.config["models"].keys():
                try:
                    all_results[name] = evaluator.evaluate_model_json(name, dataset_name, k_values, max_samples)
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    all_results[name] = {"error": str(e)}
            return all_results
    else:
        if model_name:
            return evaluator.evaluate_model(model_name, dataset_name, k_values, max_samples)
        else:
            return evaluator.evaluate_all_models(dataset_name, k_values, max_samples)

