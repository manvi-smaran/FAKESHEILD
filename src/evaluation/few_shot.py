from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import yaml

from src.models import get_model
from src.data.dataset_loader import create_dataset, DeepfakeDataset
from src.utils.prompts import build_few_shot_prompt, FEW_SHOT_EXAMPLE_REAL, FEW_SHOT_EXAMPLE_FAKE
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
        examples = dataset.get_stratified_samples(n_per_class=k // 2)
        example_indices = list(range(len(examples)))
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
            
            for idx, (image, label, manip_type) in enumerate(tqdm(dataset, desc=f"k={k}")):
                if idx in example_indices:
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
):
    evaluator = FewShotEvaluator(config_path=config_path)
    
    if model_name:
        return evaluator.evaluate_model(model_name, dataset_name, k_values, max_samples)
    else:
        return evaluator.evaluate_all_models(dataset_name, k_values, max_samples)
