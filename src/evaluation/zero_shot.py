from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
import json
import yaml

from src.models import get_model
from src.data.dataset_loader import create_dataset
from src.utils.prompts import get_prompt
from src.evaluation.metrics import (
    parse_prediction,
    extract_confidence,
    compute_metrics,
    compute_per_manipulation_metrics,
    format_results,
)


class ZeroShotEvaluator:
    
    def __init__(
        self,
        config_path: str = "configs/model_configs.yaml",
        prompt_type: str = "zero_shot_binary",
        results_dir: str = "results",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.prompt = get_prompt(prompt_type)
        self.prompt_type = prompt_type
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        model_config = self.config["models"][model_name]
        model = get_model(model_name, model_config)
        
        print(f"Loading model: {model_config['name']}...")
        model.load_model()
        
        dataset = create_dataset(self.config, dataset_name, max_samples)
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Stats: {dataset.get_stats()}")
        
        predictions = []
        confidences = []
        labels = []
        manipulation_types = []
        raw_responses = []
        
        for image, label, manip_type in tqdm(dataset, desc=f"Evaluating {model_name}"):
            response = model.predict(image, self.prompt)
            
            pred = parse_prediction(response)
            conf = extract_confidence(response)
            
            predictions.append(pred)
            confidences.append(conf)
            labels.append(label)
            manipulation_types.append(manip_type)
            raw_responses.append(response)
        
        model.cleanup()
        
        metrics = compute_metrics(predictions, labels, confidences)
        per_manip_metrics = compute_per_manipulation_metrics(
            predictions, labels, manipulation_types
        )
        
        results = {
            "model": model_name,
            "model_full_name": model_config["name"],
            "dataset": dataset_name,
            "prompt_type": self.prompt_type,
            "num_samples": len(dataset),
            "metrics": metrics,
            "per_manipulation": per_manip_metrics,
        }
        
        if save_predictions:
            results["predictions"] = {
                "pred": predictions,
                "label": labels,
                "confidence": confidences,
                "manipulation_type": manipulation_types,
                "raw_response": raw_responses,
            }
        
        print(format_results(metrics, model_config["name"], dataset_name, "Zero-Shot"))
        
        return results
    
    def evaluate_all_models(
        self,
        dataset_name: str,
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
                    max_samples,
                    save_predictions=True,
                )
                all_results[model_name] = results
            except Exception as e:
                import traceback
                print(f"Error evaluating {model_name}: {e}")
                traceback.print_exc()
                all_results[model_name] = {"error": str(e)}
        
        self._save_results(all_results, dataset_name, "zero_shot")
        
        return all_results
    
    def _save_results(
        self, 
        results: Dict[str, Any], 
        dataset_name: str,
        eval_type: str,
    ):
        output_path = self.results_dir / f"{eval_type}_{dataset_name}_results.json"
        
        serializable_results = {}
        for model_name, model_results in results.items():
            if "predictions" in model_results:
                model_results = model_results.copy()
                preds = model_results.pop("predictions")
                model_results["predictions_summary"] = {
                    "total": len(preds["pred"]),
                    "valid": sum(1 for p in preds["pred"] if p != -1),
                    "predicted_fake": sum(1 for p in preds["pred"] if p == 1),
                    "predicted_real": sum(1 for p in preds["pred"] if p == 0),
                }
            serializable_results[model_name] = model_results
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def run_zero_shot_evaluation(
    model_name: Optional[str] = None,
    dataset_name: str = "celebdf",
    max_samples: Optional[int] = None,
    prompt_type: str = "zero_shot_binary",
    config_path: str = "configs/model_configs.yaml",
):
    evaluator = ZeroShotEvaluator(
        config_path=config_path,
        prompt_type=prompt_type,
    )
    
    if model_name:
        return evaluator.evaluate_model(model_name, dataset_name, max_samples)
    else:
        return evaluator.evaluate_all_models(dataset_name, max_samples)
