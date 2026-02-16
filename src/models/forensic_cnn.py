"""
Forensic CNN Module for Deepfake Detection

Lightweight CNN that extracts forensic features (frequency anomalies,
blending artifacts, texture patterns) from images. These features are
projected into the VLM's token embedding space and prepended to the
VLM input, giving it forensic context it normally cannot detect.

Architecture:
  Image (224x224) → 3 Conv blocks → AdaptiveAvgPool → FC → Projection
  Output: num_tokens × hidden_size tensor (injectable as VLM tokens)

Usage:
  cnn = ForensicCNN(hidden_size=1536, num_tokens=4)
  forensic_tokens = cnn(pixel_tensor)  # [B, 4, 1536]
"""

import torch
import torch.nn as nn
import torchvision.transforms as T


class ForensicCNN(nn.Module):
    """
    Lightweight CNN forensic feature extractor.
    
    Extracts sub-pixel manipulation artifacts and projects them
    into the VLM's hidden dimension as virtual forensic tokens.
    
    Args:
        hidden_size: VLM hidden dimension (1536 for Qwen2-VL 2B)
        num_tokens: Number of forensic tokens to generate (default: 4)
        input_size: CNN input resolution (default: 224)
    """
    
    def __init__(self, hidden_size: int = 1536, num_tokens: int = 4, input_size: int = 224):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.input_size = input_size
        
        # =====================================================================
        # Feature Extraction Layers
        # 3 conv blocks: each = Conv2d → BatchNorm → ReLU → MaxPool
        # Designed to capture progressively higher-level forensic patterns
        # =====================================================================
        self.features = nn.Sequential(
            # Block 1: Low-level edges and noise patterns (3 → 32 channels)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112
            
            # Block 2: Mid-level blending and texture artifacts (32 → 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56
            
            # Block 3: High-level manipulation signatures (64 → 128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # → 4×4 regardless of input size
        )
        
        # =====================================================================
        # Projection Layer
        # Maps CNN features (128*4*4 = 2048) → num_tokens × hidden_size
        # This is the bridge between CNN and VLM embedding spaces
        # =====================================================================
        cnn_feature_dim = 128 * 4 * 4  # 2048
        self.projector = nn.Sequential(
            nn.Linear(cnn_feature_dim, hidden_size * num_tokens),
            nn.LayerNorm(hidden_size * num_tokens),
        )
        
        # =====================================================================
        # Preprocessing Transform
        # Normalizes image to CNN-friendly format
        # =====================================================================
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize CNN weights with small values to avoid disrupting VLM at start."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Small init so forensic tokens start near zero → minimal disruption to VLM
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def preprocess(self, pil_image):
        """Convert PIL Image to CNN input tensor."""
        return self.transform(pil_image)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract forensic features and project to VLM token space.
        
        Args:
            x: Image tensor [B, 3, H, W] (preprocessed)
            
        Returns:
            Forensic token embeddings [B, num_tokens, hidden_size]
        """
        # Extract CNN features
        features = self.features(x)            # [B, 128, 4, 4]
        features = features.flatten(1)          # [B, 2048]
        
        # Project to VLM token space
        projected = self.projector(features)    # [B, num_tokens * hidden_size]
        
        # Reshape to token format
        tokens = projected.view(-1, self.num_tokens, self.hidden_size)  # [B, num_tokens, hidden_size]
        
        return tokens
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
