from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
from PIL import Image
import random


class DeepfakeDataset:
    
    def __init__(
        self, 
        root_path: str,
        dataset_type: str,
        manipulation_types: Optional[List[str]] = None,
        compression: str = "c23",
        frame_format: str = "jpg",
        max_samples: Optional[int] = None,
        seed: int = 42,
        celebdf_path: Optional[str] = None,
        faceforensics_path: Optional[str] = None,
    ):
        self.root_path = Path(root_path)
        self.dataset_type = dataset_type
        self.manipulation_types = manipulation_types or []
        self.compression = compression
        self.frame_format = frame_format
        self.max_samples = max_samples
        self.seed = seed
        self.celebdf_path = Path(celebdf_path) if celebdf_path else None
        self.faceforensics_path = Path(faceforensics_path) if faceforensics_path else None
        
        self.samples: List[Tuple[Path, int, str]] = []
        self._load_samples()
    
    def _load_samples(self):
        random.seed(self.seed)
        
        if self.dataset_type == "celebdf":
            self._load_celebdf()
        elif self.dataset_type == "faceforensics":
            self._load_faceforensics()
        elif self.dataset_type == "combined":
            self._load_combined()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)

    def _load_combined(self):
        """Load both Celeb-DF and FaceForensics samples."""
        if not self.celebdf_path or not self.faceforensics_path:
            raise ValueError("Both celebdf_path and faceforensics_path must be provided for combined dataset")
            
        # Load Celeb-DF (temporarily swap root_path)
        original_root = self.root_path
        self.root_path = self.celebdf_path
        self.frame_format = "jpg" # Celeb-DF is jpg
        self._load_celebdf()
        celeb_count = len(self.samples)
        
        # Load FaceForensics
        self.root_path = self.faceforensics_path
        self.frame_format = "png" # FaceForensics is png (usually) - verified in loader it looks for multiple extensions
        
        # Set default manipulation types for FF if not set
        if not self.manipulation_types:
             self.manipulation_types = ["original", "Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
             
        self._load_faceforensics()
        ff_count = len(self.samples) - celeb_count
        
        # Restore root path (though not strictly used after loading)
        self.root_path = original_root
        print(f"[DATA] Combined loaded: {celeb_count} from Celeb-DF, {ff_count} from FaceForensics")
    
    def _load_celebdf(self):
        real_dirs = [
            self.root_path / "Celeb-real",
            self.root_path / "YouTube-real",
        ]
        fake_dir = self.root_path / "Celeb-synthesis"
        
        for real_dir in real_dirs:
            if real_dir.exists():
                for video_dir in real_dir.iterdir():
                    if video_dir.is_dir():
                        frames = list(video_dir.glob(f"*.{self.frame_format}"))
                        for frame in frames:
                            self.samples.append((frame, 0, "real"))
        
        if fake_dir.exists():
            for video_dir in fake_dir.iterdir():
                if video_dir.is_dir():
                    frames = list(video_dir.glob(f"*.{self.frame_format}"))
                    for frame in frames:
                        self.samples.append((frame, 1, "deepfake"))
        
        if not self.samples:
            self._load_celebdf_flat()
    
    def _load_celebdf_flat(self):
        for subdir in self.root_path.iterdir():
            if not subdir.is_dir():
                continue
            
            subdir_lower = subdir.name.lower()
            
            if "real" in subdir_lower:
                label = 0
                manip_type = "real"
            elif "fake" in subdir_lower or "synthesis" in subdir_lower:
                label = 1
                manip_type = "deepfake"
            else:
                continue
            
            for frame in subdir.glob(f"**/*.{self.frame_format}"):
                self.samples.append((frame, label, manip_type))
    
    def _load_faceforensics(self):
        manip_map = {
            "Deepfakes": 1,
            "Face2Face": 1,
            "FaceSwap": 1,
            "FaceShifter": 1,
            "NeuralTextures": 1,
            "DeepFakeDetection": 1,
            "original_sequences": 0,
            "original": 0,
            "real": 0,
            "fake": 1,
        }
        
        types_to_load = self.manipulation_types if self.manipulation_types else list(manip_map.keys())
        
        for manip_type in types_to_load:
            if manip_type in ["original_sequences", "original"]:
                possible_dirs = [
                    self.root_path / "original_sequences" / "youtube" / self.compression / "frames",
                    self.root_path / "original",
                    self.root_path / "real",
                    self.root_path / "Original",
                ]
            else:
                possible_dirs = [
                    self.root_path / "manipulated_sequences" / manip_type / self.compression / "frames",
                    self.root_path / manip_type,
                    self.root_path / "fake",
                    self.root_path / "Fake",
                    self.root_path / "Deepfakes",
                ]
            
            for manip_dir in possible_dirs:
                if manip_dir.exists():
                    label = manip_map.get(manip_type, 1 if "fake" in manip_type.lower() or "deep" in manip_type.lower() else 0)
                    for ext in [self.frame_format, "jpg", "png", "jpeg"]:
                        for frame in manip_dir.glob(f"**/*.{ext}"):
                            self.samples.append((frame, label, manip_type))
                    break
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        frame_path, label, manip_type = self.samples[idx]
        image = Image.open(frame_path).convert("RGB")
        return image, label, manip_type
    
    def __iter__(self) -> Iterator[Tuple[Image.Image, int, str]]:
        for idx in range(len(self)):
            yield self[idx]
    
    def get_stratified_samples(
        self, n_per_class: int, return_indices: bool = False
    ) -> List[Tuple[Image.Image, int, str]]:
        """
        Get stratified samples from the dataset.
        
        Args:
            n_per_class: Number of samples per class
            return_indices: If True, return (samples, indices) tuple
            
        Returns:
            If return_indices=False: List of (image, label, manip_type) tuples
            If return_indices=True: Tuple of (samples_list, indices_list)
        """
        # Track indices along with samples
        real_indices = [i for i, s in enumerate(self.samples) if s[1] == 0]
        fake_indices = [i for i, s in enumerate(self.samples) if s[1] == 1]
        
        random.seed(self.seed)
        
        selected_real_idx = random.sample(real_indices, min(n_per_class, len(real_indices)))
        selected_fake_idx = random.sample(fake_indices, min(n_per_class, len(fake_indices)))
        
        selected_indices = selected_real_idx + selected_fake_idx
        
        result = []
        for idx in selected_indices:
            frame_path, label, manip_type = self.samples[idx]
            image = Image.open(frame_path).convert("RGB")
            result.append((image, label, manip_type))
        
        if return_indices:
            return result, selected_indices
        return result
    
    def get_stats(self) -> Dict[str, int]:
        stats = {"total": len(self.samples), "real": 0, "fake": 0}
        manip_stats: Dict[str, int] = {}
        
        for _, label, manip_type in self.samples:
            if label == 0:
                stats["real"] += 1
            else:
                stats["fake"] += 1
            
            manip_stats[manip_type] = manip_stats.get(manip_type, 0) + 1
        
        stats["by_manipulation"] = manip_stats
        return stats


def create_dataset(config: dict, dataset_name: str, max_samples: Optional[int] = None) -> DeepfakeDataset:
    dataset_config = config["datasets"][dataset_name]
    
    return DeepfakeDataset(
        root_path=dataset_config.get("root_path", ""), # Optional for combined
        dataset_type=dataset_name,
        manipulation_types=dataset_config.get("manipulation_types"),
        compression=dataset_config.get("compression", "c23"),
        frame_format=dataset_config.get("frame_format", "jpg"),
        max_samples=max_samples,
        celebdf_path=dataset_config.get("celebdf_path"),
        faceforensics_path=dataset_config.get("faceforensics_path"),
    )
